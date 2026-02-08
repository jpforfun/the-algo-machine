# execution/execution_engine.py
"""
Broker-safe execution engine using Kite Connect.
Handles order placement, modification, and lifecycle management.

This module is designed to be defensive:
1. Re-checks risk (spread, liquidity) at the point of execution.
2. Robustly handles partial fills in pegged orders.
3. Implements synchronous-like fallback to avoid double-filling.
4. Reconciles state on startup.
"""

import logging
import time
import threading
import uuid
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone, timedelta

from config.config import get_settings
from state.state_manager import StateManager, Order, OrderSide, OrderStatus, RiskFlag
from risk.risk_manager import RiskManager
from features.microstructure_features import (
    MicrostructureSnapshot,
    MicrostructureFeatureEngine
)

# Try importing KiteConnect, handle missing dependency for testing
try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import (
        InputException, NetworkException, DataException, 
        OrderException, TokenException
    )
except ImportError:
    # Pseudo-mock for environments without kiteconnect installed
    class KiteConnect: pass
    class InputException(Exception): pass
    class NetworkException(Exception): pass
    class DataException(Exception): pass
    class OrderException(Exception): pass
    class TokenException(Exception): pass
    logging.warning("KiteConnect not installed. ExecutionEngine will fail if instantiated without mock.")

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Handles order execution with Zerodha Kite Connect.
    
    Implements:
    - Marketable Limit Orders: Safety buffer on limit price for market-like fills.
    - Pegged Limit Orders: Chases best price with strict TTL and modification limits.
    - Rate Limiting: Respects broker API limits.
    - Error Recovery: Graceful fallback to marketable limits on failures.
    - Defensive Risk: Re-verifies market conditions before every action.

    Note: ExecutionEngine is intentionally thread-based to isolate broker I/O 
    from the asyncio event loop.
    """
    
    def __init__(self, 
                 kite: KiteConnect, 
                 state_manager: StateManager, 
                 risk_manager: RiskManager,
                 feature_engine: MicrostructureFeatureEngine):
        """
        Initialize the Execution Engine.
        
        Args:
            kite: Authenticated KiteConnect instance
            state_manager: Shared trading state
            risk_manager: Risk management gatekeeper
            feature_engine: Engine for fetching latest market state
        """
        self.kite = kite
        self.state_manager = state_manager
        self.risk_manager = risk_manager
        self.feature_engine = feature_engine
        self.settings = get_settings()
        
        self._lock = threading.RLock()
        self._rate_limit_lock = threading.Lock()
        self._last_order_time = 0.0
        
        # Track pegged orders in memory for monitoring
        # Format: { local_id: { broker_id, symbol, side, original_qty, remaining_qty, price, ... } }
        self._active_pegged_orders: Dict[str, Dict[str, Any]] = {} 
        self._broker_failure_count = 0  # To track consecutive broker errors (FIXED HARDENING #2)
        self._stop_event = threading.Event()
        
        # Minimum tick size from config
        self._tick_size = self.settings.execution_tick_size
        self._inv_tick = 1.0 / self._tick_size
        
        # Startup Reconciliation
        self.reconcile_state()
        
        # Monitoring thread for pegged orders
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def reconcile_state(self):
        """
        Reconciles in-memory state with broker and DB on startup.
        Identifies unmanaged open orders and either rebuilds tracking or cancels them.
        """
        try:
            logger.info("Starting Execution Engine state reconciliation...")
            # 1. Fetch all open orders from broker
            broker_orders = self.kite.orders()
            open_broker_orders = [o for o in broker_orders if o['status'] in ['OPEN', 'MODIFY PENDING', 'TRIGGER PENDING']]
            
            # 2. Get our DB's view of open orders
            db_open_orders = self.state_manager.get_open_orders()
            db_map = {o.broker_order_id: o for o in db_open_orders if o.broker_order_id}
            
            for b_order in open_broker_orders:
                b_id = str(b_order['order_id'])
                if b_id in db_map:
                    order_rec = db_map[b_id]
                    # If it was a pegged order (based on tags), rebuild tracking
                    if order_rec.tags and order_rec.tags.get("type") == "pegged":
                        logger.info(f"Rebuilding tracking for pegged order {order_rec.order_id} (Broker: {b_id})")
                        with self._lock:
                            self._active_pegged_orders[order_rec.order_id] = {
                                "broker_id": b_id,
                                "symbol": order_rec.symbol,
                                "side": order_rec.side.value if hasattr(order_rec.side, 'value') else order_rec.side,
                                "original_qty": order_rec.quantity,
                                "remaining_qty": order_rec.quantity - order_rec.filled_quantity,
                                "last_filled_seen": order_rec.filled_quantity,
                                "price": order_rec.price,
                                "start_time": time.time(), # Reset TTL on restart to be safe
                                "modifications": order_rec.tags.get("mods", 0),
                                "tag": order_rec.tags.get("tag", "algo")
                            }
                else:
                    # Unmanaged order found at broker - cancel for safety
                    logger.warning(f"Unmanaged open order {b_id} found at broker. Cancelling for safety.")
                    try:
                        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=b_id)
                    except Exception as e:
                        logger.error(f"Failed to cancel unmanaged order {b_id}: {e}")
            
            logger.info("Execution Engine reconciliation complete.")
        except Exception as e:
            logger.critical(f"Critical error during Execution Engine reconciliation: {e}")

    def _apply_rate_limit(self):
        """Ensures compliance with Zerodha order placement rate limits."""
        with self._rate_limit_lock:
            now = time.time()
            delay = (self.settings.order_placement_delay_ms / 1000.0)
            elapsed = now - self._last_order_time
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last_order_time = time.time()

    def _normalize_side(self, side: Union[str, OrderSide]) -> OrderSide:
        """Ensures side is always an OrderSide enum."""
        if isinstance(side, OrderSide):
            return side
        return OrderSide[side.upper()]

    def place_order(self, 
                    symbol: str, 
                    side: str, 
                    quantity: int, 
                    price: Optional[float] = None,
                    snapshot: Optional[MicrostructureSnapshot] = None,
                    tag: str = "algo",
                    is_pegged: bool = False,
                    urgent: bool = False,
                    alpha: float = 0.0,
                    rank: float = 0.0) -> Optional[str]:
        """
        Main entry point for placing orders with defensive risk checks.
        
        Args:
            symbol: Exchange tradingsymbol
            side: 'BUY' or 'SELL'
            quantity: Quantity to trade
            price: Target price (None for market-like execution)
            snapshot: Current market state for re-checking risk
            tag: Metadata tag
            is_pegged: Enabled passive chasing logic
            urgent: If True, bypasses pegging for immediate fill
            
        Returns:
            Local order ID if successful
        """
        if quantity <= 0:
            return None
            
        # 0. Sovereignty Check: Block execution if system error flag is set (FIXED HARDENING #1)
        if self.state_manager.is_trading_halted():
            logger.critical("SYSTEM_ERROR flag set or trading halted. Execution blocked.")
            return None

        # 0.5 Dry-Run Guard (🛡️ Safety Improvement)
        norm_side = self._normalize_side(side)
        if self.settings.dry_run_mode:
            logger.info(f"[DRY-RUN] Simulating order placement: {norm_side} {quantity} {symbol} (is_pegged={is_pegged})")
            fake_id = f"SIM_{uuid.uuid4().hex[:10]}"
            # Save a record of the simulated order
            order_rec = Order(
                order_id=fake_id,
                symbol=symbol,
                side=norm_side,
                quantity=quantity,
                filled_quantity=quantity,
                price=price or (snapshot.last_price if snapshot else 0.0),
                trigger_price=None,
                status=OrderStatus.COMPLETE,
                order_timestamp=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                tags={
                    "tag": tag, 
                    "type": "dry_run",
                    "alpha": round(alpha, 4),
                    "rank": round(rank, 2)
                }
            )
            self.state_manager.save_order(order_rec)
            return fake_id
            
        # 1. Defensive Execution-side Risk Check
        # Re-fetch latest snapshot independently to avoid trusting stale data from Alpha
        current_snap = self.feature_engine.get_snapshot_by_symbol(symbol) or snapshot
        
        if not self.risk_manager.approve_trade(symbol, norm_side.value, quantity, current_snap, is_urgent=urgent):
            logger.warning(f"Execution-side Risk Reject for {symbol}: Market conditions shifted.")
            return None
            
        local_id = f"ORD_{uuid.uuid4().hex[:12]}"
        
        # 2. Decision logic for execution style
        # Upgrade snapshot reference for internal placement methods
        exec_snap = current_snap 
        if is_pegged and not urgent and self.settings.use_pegged_orders:
            return self._place_pegged_order(local_id, symbol, norm_side, quantity, tag, exec_snap, alpha, rank)
        else:
            return self._place_marketable_limit(local_id, symbol, norm_side, quantity, price, tag, exec_snap, alpha, rank)

    def _place_marketable_limit(self, local_id: str, symbol: str, side: OrderSide, 
                                quantity: int, price: Optional[float], tag: str,
                                snapshot: Optional[MicrostructureSnapshot],
                                alpha: float = 0.0, rank: float = 0.0) -> Optional[str]:
        """Places a limit order with a buffer for market-like fills."""
        try:
            # 1. Price discovery
            exec_price = price
            if exec_price is None or exec_price == 0:
                if snapshot:
                    ltp = snapshot.last_price
                else:
                    quote = self.kite.quote(f"NSE:{symbol}")
                    ltp = quote[f"NSE:{symbol}"]["last_price"]
                
                buffer_pct = self.settings.execution_marketable_limit_buffer_bps / 10000.0
                if side == OrderSide.BUY:
                    exec_price = ltp * (1 + buffer_pct)
                else:
                    exec_price = ltp * (1 - buffer_pct)
            
            # 2. Final normalization
            exec_price = self._round_to_tick(exec_price)
            
            # 3. Save initial state
            order_rec = Order(
                order_id=local_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0,
                price=exec_price,
                trigger_price=None,
                status=OrderStatus.PENDING,
                order_timestamp=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                tags={
                    "tag": tag, 
                    "type": "marketable_limit",
                    "alpha": round(alpha, 4),
                    "rank": round(rank, 2)
                }
            )
            self.state_manager.save_order(order_rec)
            
            # 4. API Call with specific exception handling
            self._apply_rate_limit()
            broker_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == OrderSide.BUY else self.kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS if self.settings.trading_mode != "LIVE" else self.kite.PRODUCT_MIS, # Forced to MIS safe default
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=exec_price,
                tag=tag
            )
            
            self.state_manager.update_order_status(local_id, OrderStatus.OPEN, broker_order_id=str(broker_id))
            logger.info(f"Marketable Limit placed: {symbol} {side.value} {quantity} @ {exec_price} "
                        f"(Local: {local_id}, Broker: {broker_id})")
            return local_id

        except NetworkException as e:
            logger.error(f"Network error placing order for {symbol}: {e}. Retry handled by management layer.")
            self.state_manager.update_order_status(local_id, OrderStatus.REJECTED)
            return None
        except TokenException as e:
            logger.critical(f"CRITICAL: Token Expired during order placement. Halting system: {e}")
            self.state_manager.set_risk_flag(RiskFlag.SYSTEM_ERROR, {"reason": "TokenException during placement"})
            return None
        except Exception as e:
            logger.error(f"Failed to place marketable limit for {symbol}: {e}")
            self.state_manager.update_order_status(local_id, OrderStatus.REJECTED)
            return None

    def _place_pegged_order(self, local_id: str, symbol: str, side: OrderSide, 
                            quantity: int, tag: str, snapshot: Optional[MicrostructureSnapshot],
                            alpha: float = 0.0, rank: float = 0.0) -> Optional[str]:
        """Places a passive limit order and registers it for chasing."""
        try:
            # 1. Passive Price Discovery
            best_passive = 0.0
            if snapshot:
                best_passive = snapshot.best_bid if side == OrderSide.BUY else snapshot.best_ask
            else:
                quote = self.kite.quote(f"NSE:{symbol}")
                q_data = quote[f"NSE:{symbol}"]
                best_passive = q_data["depth"]["buy"][0]["price"] if side == OrderSide.BUY else q_data["depth"]["sell"][0]["price"]
            
            if best_passive == 0.0:
                 logger.error(f"Could not determine passive price for {symbol}")
                 return None

            # 2. Save and Place
            order_rec = Order(
                order_id=local_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0,
                price=best_passive,
                trigger_price=None,
                status=OrderStatus.PENDING,
                order_timestamp=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                tags={
                    "tag": tag, 
                    "type": "pegged", 
                    "mods": 0,
                    "alpha": round(alpha, 4),
                    "rank": round(rank, 2)
                }
            )
            self.state_manager.save_order(order_rec)
            
            self._apply_rate_limit()
            broker_id = str(self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == OrderSide.BUY else self.kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=best_passive,
                tag=tag
            ))
            
            self.state_manager.update_order_status(local_id, OrderStatus.OPEN, broker_order_id=broker_id)
            
            # 3. Register for monitoring
            with self._lock:
                self._active_pegged_orders[local_id] = {
                    "broker_id": broker_id,
                    "symbol": symbol,
                    "side": side,
                    "original_qty": quantity,
                    "remaining_qty": quantity,
                    "last_filled_seen": 0,
                    "price": best_passive,
                    "start_time": time.time(),
                    "modifications": 0,
                    "tag": tag
                }
            
            logger.info(f"Pegged Order started: {symbol} {side.value} @ {best_passive} (Local: {local_id}, Broker: {broker_id})")
            return local_id

        except Exception as e:
            logger.error(f"Failed to start pegged order for {symbol}: {e}")
            self.state_manager.update_order_status(local_id, OrderStatus.REJECTED)
            return None

    def _monitor_loop(self):
        """Background loop to manage pegged order lifecycle with batched status checks."""
        while not self._stop_event.is_set():
            try:
                # Halt Enforcement: Absolute block (🛡️ Safety Improvement)
                if self.state_manager.is_trading_halted():
                    time.sleep(5)
                    continue

                # 1. Collect all local IDs
                active_ids = []
                with self._lock:
                    active_ids = list(self._active_pegged_orders.keys())
                
                if not active_ids:
                    time.sleep(1)
                    continue
                
                # 2. BATCHED status and quote updates
                # Kite REST doesn't have a single "multi-order status by ID", but we can get ALL orders once.
                # Scalable for 1-100 orders instead of 1 call PER order.
                try:
                    all_orders = {str(o['order_id']): o for o in self.kite.orders()}
                    self._broker_failure_count = 0  # Reset on success
                except Exception as e:
                    self._broker_failure_count += 1
                    logger.error(f"Broker error in monitor loop ({self._broker_failure_count}/3): {e}")
                    
                    if self._broker_failure_count >= 3:
                        logger.critical("Consecutive broker failures detected. Triggering emergency fallback.")
                        self.state_manager.set_risk_flag(
                            RiskFlag.SYSTEM_ERROR, 
                            {"reason": "Consecutive broker failures in execution monitor"}
                        )
                        # Emergency fallback for all active pegged orders
                        for oid in active_ids:
                            self._fallback_to_marketable(oid)
                    
                    time.sleep(5)
                    continue
                
                symbols = [self._active_pegged_orders[oid]["symbol"] for oid in active_ids]
                unique_symbols = list(set(symbols))
                instruments = [f"NSE:{s}" for s in unique_symbols]
                quotes = self.kite.quote(instruments)
                
                # 3. Process each tracked order
                for local_id in active_ids:
                    self._check_and_modify_pegged(local_id, all_orders, quotes)
                
                # Dynamic delay based on number of active orders? Default is fine.
                time.sleep(self.settings.execution_order_poll_interval_ms / 1000.0)

            except Exception as e:
                logger.error(f"Error in execution monitor loop: {e}")
                time.sleep(5)

    def _check_and_modify_pegged(self, local_id: str, all_broker_orders: Dict, quotes: Dict):
        """
        Evaluates a single pegged order using batched data.
        Correctly tracks remaining_qty to avoid double-filling.
        """
        with self._lock:
            if local_id not in self._active_pegged_orders:
                return
            
            p = self._active_pegged_orders[local_id]
            broker_id = p["broker_id"]
            
            # Sync status and filled quantity
            b_order = all_broker_orders.get(broker_id)
            if not b_order:
                # Order not found in recent history - might be older than poll cycle or broker transient error
                return
                
            status = b_order.get("status")
            filled = b_order.get("filled_quantity", 0)
            
            # Partial Fill Mitigation (🛡️ Safety Improvement)
            if filled > p["last_filled_seen"]:
                logger.debug(f"Partial fill detected for {local_id}: {filled} > {p['last_filled_seen']}. Skipping modification this cycle.")
                p["last_filled_seen"] = filled
                p["remaining_qty"] = p["original_qty"] - filled
                self.state_manager.update_order_status(local_id, OrderStatus.PARTIAL_FILL, filled_quantity=filled)
                return

            p["remaining_qty"] = p["original_qty"] - filled
            
            # Sync to DB
            our_status = OrderStatus.OPEN
            if status == "COMPLETE": our_status = OrderStatus.COMPLETE
            elif status == "CANCELLED": our_status = OrderStatus.CANCELLED
            elif status == "REJECTED": our_status = OrderStatus.REJECTED
            elif filled > 0: our_status = OrderStatus.PARTIAL_FILL
            
            self.state_manager.update_order_status(local_id, our_status, filled_quantity=filled)
            
            if our_status in [OrderStatus.COMPLETE, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                with self._lock:
                    self._active_pegged_orders.pop(local_id, None)
                return

            # 2. Defensive check: Is market still tradeable?
            symbol = p["symbol"]
            side = p["side"]
            
            # Fetch latest snapshot for re-verifying risk during chase
            current_snap = self.feature_engine.get_snapshot_by_symbol(symbol)
            
            if p["remaining_qty"] <= 0:
                with self._lock:
                    self._active_pegged_orders.pop(local_id, None)
                return

            if not self.risk_manager.approve_trade(symbol, side.value, p["remaining_qty"], current_snap):
                logger.warning(f"Market shifted against pegged order {local_id}. Triggering fallback.")
                self._fallback_to_marketable(local_id)
                return

            # 3. Check modification limits/TTL
            elapsed = time.time() - p["start_time"]
            if elapsed > self.settings.execution_pegged_ttl_sec or \
               p["modifications"] >= self.settings.execution_pegged_max_modifications:
                logger.warning(f"Pegged order {local_id} hit limits. Falling back.")
                self._fallback_to_marketable(local_id)
                return

            # 4. Check for price chase
            q_data = quotes.get(f"NSE:{symbol}")
            if not q_data or "depth" not in q_data:
                return
                
            best_passive = q_data["depth"]["buy"][0]["price"] if side == OrderSide.BUY else q_data["depth"]["sell"][0]["price"]
            price_diff = abs(p["price"] - best_passive)
            if price_diff > (self._tick_size * 0.9): # Slightly less than a tick to avoid float issues
                self._modify_pegged_to_price(local_id, best_passive)

    def _modify_pegged_to_price(self, local_id: str, new_price: float):
        """Modifies order while respecting rate limits."""
        p = self._active_pegged_orders[local_id]
        try:
            self._apply_rate_limit()
            self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=p["broker_id"],
                price=new_price
            )
            p["price"] = new_price
            p["modifications"] += 1
            logger.info(f"Pegged order {local_id} (Broker: {p['broker_id']}) chased to {new_price} (#mod {p['modifications']})")
        except Exception as e:
            logger.error(f"Failed to modify pegged order {local_id}: {e}")

    def _fallback_to_marketable(self, local_id: str):
        """
        Synchronously-equivalent fallback to prevent double-fills.
        Cancel -> Poll/Wait -> Recompute -> Place Marketable Limit.
        """
        p = self._active_pegged_orders.get(local_id)
        if not p: return

        broker_id = p["broker_id"]
        logger.info(f"Executing fallback for {local_id} (Broker: {broker_id})")

        try:
            # 1. Request Cancellation and Track intent (🛡️ Safety Improvement)
            self._apply_rate_limit()
            self.state_manager.update_order_status(local_id, OrderStatus.CANCEL_PENDING)
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=broker_id)
            
            # 2. Wait for final status and recompute remaining balance accurately
            # We poll a few times with small delay
            remaining = 0
            for i in range(5):
                time.sleep(0.5)
                # Fetch order directly for highest precision in fallback
                b_order = self.kite.order_history(broker_id)[-1]
                status = b_order["status"]
                filled = b_order["filled_quantity"]
                total = b_order["quantity"]
                
                if status in ["CANCELLED", "COMPLETE", "REJECTED"]:
                    remaining = total - filled
                    break
                logger.debug(f"Waiting for {broker_id} cancel confirmation... current status: {status}")
            
            # 3. Remove from tracking BEFORE placing new order to prevent race
            with self._lock:
                self._active_pegged_orders.pop(local_id, None)

            # 4. Place fallback marketable limit for EXACT remaining portion
            if remaining > 0:
                logger.info(f"Placing fallback for remaining {remaining} of {local_id}")
                self._place_marketable_limit(
                    f"{local_id}_FB", p["symbol"], p["side"], remaining, None, p["tag"], None
                )
            else:
                logger.info(f"Pegged order {local_id} filled completely during cancel attempt.")

        except Exception as e:
            logger.critical(f"CRITICAL: Fallback failed for {local_id}: {e}. Potential exposure risk!")

    def _round_to_tick(self, price: float) -> float:
        """
        Rounds price to nearest configured tick size.
        """
        return round(price * self._inv_tick) / self._inv_tick

    def stop(self):
        """Gracefully stops the execution monitor."""
        logger.info("Stopping Execution Engine...")
        self._stop_event.set()
        if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("Execution Engine stopped.")
