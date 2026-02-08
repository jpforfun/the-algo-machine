# core/engine.py
"""
Central Trading Engine Orchestrator.
Wires together TickerService, FeatureEngine, Ranker, AlphaEngine, and ExecutionEngine.
Focus: Capital Safety, Decoupling, and Performance.
"""

import asyncio
import logging
import signal
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta

from config.config import get_settings
from state.state_manager import StateManager, OrderStatus, OrderSide, RiskFlag, RiskFlag
from data.ticker_service import TickerService, create_ticker_service
from features.microstructure_features import MicrostructureFeatureEngine, MicrostructureSnapshot
from strategy.cross_sectional_ranker import CrossSectionalRanker
from alpha.alpha_engine import AlphaEngine
from risk.risk_manager import RiskManager
from execution.execution_engine import ExecutionEngine

# Mock KiteConnect if not installed (for dev/test)
try:
    from kiteconnect import KiteConnect
except ImportError:
    class KiteConnect:
        def __init__(self, *args, **kwargs): pass
    logging.warning("KiteConnect not installed. TradingEngine using mock.")

logger = logging.getLogger(__name__)

# IST Timezone for NSE
IST = timezone(timedelta(hours=5, minutes=30))

@dataclass
class TradeIntent:
    """Decoupled record of a trading decision from AlphaEngine."""
    symbol: str
    side: str
    quantity: int
    snapshot: MicrostructureSnapshot
    alpha: float
    rank: float
    is_pegged: bool
    urgent: bool
    tag: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class SignalGate:
    """
    Prevents over-execution by enforcing per-symbol cooldowns and tracking in-flight orders.
    """
    def __init__(self, cooldown_sec: int = 60):
        self._cooldown_sec = cooldown_sec
        self._last_signal_time: Dict[str, float] = {}
        self._in_flight: Set[str] = set() # Symbols with orders currently being processed by ExecutionWorker
        self._lock = threading.Lock()

    def can_proceed(self, symbol: str) -> bool:
        """Check if a new signal for this symbol is allowed."""
        with self._lock:
            if symbol in self._in_flight:
                return False
            
            last_time = self._last_signal_time.get(symbol, 0)
            if time.monotonic() - last_time < self._cooldown_sec:
                return False
                
            return True

    def mark_in_flight(self, symbol: str):
        """Register that a signal is being processed."""
        with self._lock:
            self._in_flight.add(symbol)
            self._last_signal_time[symbol] = time.monotonic()

    def clear_in_flight(self, symbol: str):
        """Unregister after execution attempt (success or failure)."""
        with self._lock:
            self._in_flight.discard(symbol)

class TradingEngine:
    """
    Main orchestrator for the trading system.
    Handles the async lifecycle of all components and wires data flows with safety gates.
    """
    
    def __init__(self, kite: Optional[KiteConnect] = None):
        self.settings = get_settings()
        self.logger = logger
        
        # 1. External Client
        self.kite = kite or self._init_kite()
        
        # 2. Strategy & Risk Layers
        self.feature_engine = MicrostructureFeatureEngine()
        self.ranker = CrossSectionalRanker()
        self.alpha_engine = AlphaEngine()
        self.state_manager = StateManager(db_path=self.settings.state_db_path)
        self.risk_manager = RiskManager(self.state_manager)
        
        # 3. Execution Layer
        self.execution = ExecutionEngine(
            self.kite, 
            self.state_manager, 
            self.risk_manager,
            self.feature_engine
        )
        
        # 4. Safety Gates
        self.signal_gate = SignalGate(cooldown_sec=self.settings.order_ttl_sec or 60)
        self.intent_queue: asyncio.Queue[TradeIntent] = asyncio.Queue(maxsize=100)

        # 4.1 Mode Enforcement (🛡️ Safety Improvement)
        if self.settings.trading_mode == "LIVE" and self.settings.dry_run_mode:
            self.logger.warning("TRADING_MODE is LIVE but DRY_RUN_MODE is True. Simulation only.")
        elif self.settings.trading_mode != "LIVE" and not self.settings.dry_run_mode:
            # Prevent accidental live trading in PAPER mode
            self.logger.error(f"Inconsistent configuration: trading_mode={self.settings.trading_mode} "
                            "but dry_run_mode is False. Forcing dry_run_mode=True.")
            # Since settings are frozen, we'd need to handle this via assertion or error
            assert self.settings.dry_run_mode is True, "Live trading requires TRADING_MODE=LIVE"
        
        # 5. Data Layer (Initialized later)
        self.ticker: Optional[TickerService] = None
        
        # 6. Lifecycle Management
        self._stop_event = asyncio.Event()
        self._shutdown_lock = asyncio.Lock()
        self._is_shutting_down = False
        self._tasks: List[asyncio.Task] = []

    def _init_kite(self) -> KiteConnect:
        """Initialize and authenticate KiteConnect."""
        kc = KiteConnect(api_key=self.settings.broker_api_key)
        kc.set_access_token(self.settings.broker_access_token)
        
        # Mandatory Session Validation (🛡️ Connection Safety)
        try:
            profile = kc.profile()
            self.logger.info(f"Kite session validated for user: {profile.get('user_name', 'Unknown')}")
        except Exception as e:
            self.logger.critical(f"Kite session validation FAILED: {e}")
            raise RuntimeError("Invalid or expired Kite session. Run 'python -m auth.zerodha_login' first.")
            
        return kc

    async def run(self):
        """Starts the engine and runs until stopped."""
        self.logger.info("Initializing Trading Engine Orchestrator...")
        
        # 0. Kill-Switch Check (🛡️ Operator Safety)
        import os
        if os.path.exists(self.settings.kill_switch_file):
            self.logger.critical(f"KILL-SWITCH DETECTED: {self.settings.kill_switch_file}. Halting before start.")
            self.state_manager.set_risk_flag(RiskFlag.TRADING_HALTED, {"reason": "Kill-switch file present"})
            raise RuntimeError(f"Trading halted: Kill-switch active ({self.settings.kill_switch_file})")

        # 1. Startup Reconciliation
        self.execution.reconcile_state()
        
        # 2. Setup Data Layer
        self.ticker = create_ticker_service(
            api_key=self.settings.broker_api_key,
            access_token=self.settings.broker_access_token
        )
        self.ticker.tick_store.add_callback(self._on_tick_received)
        
        # 3. Start Background Loops
        self._tasks.append(asyncio.create_task(self._ranking_loop()))
        self._tasks.append(asyncio.create_task(self._alpha_decision_loop()))
        self._tasks.append(asyncio.create_task(self._execution_worker()))
        self._tasks.append(asyncio.create_task(self._monitoring_loop()))
        self._tasks.append(asyncio.create_task(self._session_management_loop()))
        
        # 4. Start Ticker Service
        await self.ticker.start_async()
        
        self.logger.info("Trading Engine Orchestrator is ACTIVE.")
        
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            self.logger.info("Engine run task cancelled.")
        finally:
            await self.shutdown()

    def _on_tick_received(self, tick_data: Any):
        """Callback from TickerService to update FeatureEngine."""
        self.feature_engine.update(tick_data)

    async def _ranking_loop(self):
        """Periodically trigger cross-sectional ranking."""
        self.logger.info("Starting Ranking Loop...")
        interval = getattr(self.settings, 'rank_interval_sec', 60)
        
        while not self._stop_event.is_set():
            try:
                # Capture snapshot once for consistent ranking
                snapshots = self.feature_engine.get_all_features()
                if snapshots:
                    # Ranking update is internal to the ranker and single-threaded in this loop
                    self.ranker.update(snapshots)
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in ranking loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _alpha_decision_loop(self):
        """
        Main decision loop: Snapshot -> Rank -> Alpha.
        Pushes intents to queue instead of executing directly.
        """
        self.logger.info("Starting Alpha Decision Loop...")
        
        poll_interval = getattr(self.settings, 'alpha_poll_interval_ms', 500) / 1000.0
        
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                # 1. Critical Halt Check
                if self.state_manager.is_trading_halted():
                    self.logger.critical("TRADING HALT ACTIVE. Suspending alpha generation.")
                    await asyncio.sleep(5)
                    continue

                # 2. Check Session Gate
                if not self._is_market_open_for_trading():
                    await asyncio.sleep(1)
                    continue

                # 3. Get latest feature snapshots
                snapshots = self.feature_engine.get_all_features()
                if not snapshots:
                    await asyncio.sleep(poll_interval)
                    continue
                
                # 4. Consistency Freeze: Capture ranks once per cycle
                ranker_snapshot = dict(self.ranker._ranks)
                
                # 5. Process each instrument
                for token, snap in snapshots.items():
                    # A. Fetch rank from frozen snapshot
                    rank_res = ranker_snapshot.get(token)
                    
                    # B. Get Regime Probability (Placeholder)
                    regime_prob = 1.0 
                    
                    # C. Compute Alpha Score
                    alpha = self.alpha_engine.compute_alpha(snap, rank_res, regime_prob)
                    
                    # D. Evaluate Signal and Queue Intent
                    await self._evaluate_signal(snap, alpha)
                
                # 6. Dynamic sleep using monotonic time
                elapsed = time.monotonic() - start_time
                await asyncio.sleep(max(0, poll_interval - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alpha decision loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _evaluate_signal(self, snap: MicrostructureSnapshot, alpha: Any):
        """Translates alpha signals into TradeIntents."""
        symbol = snap.symbol
        
        # Gate Check: Cooldown and In-Flight prevention
        if not self.signal_gate.can_proceed(symbol):
            return

        positions = self.state_manager.get_open_positions(symbol)
        active_pos = positions[0] if positions else None
        
        intent = None
        
        # 1. Entry Logic
        if alpha.is_entry and not active_pos:
            side = 'BUY' if alpha.signal > 0 else 'SELL'
            # Check if we already have pending orders in DB (backup for Gate)
            if not self.state_manager.get_open_orders(symbol):
                intent = TradeIntent(
                    symbol=symbol, side=side, 
                    quantity=self.settings.default_order_quantity,
                    snapshot=snap, 
                    alpha=alpha.score,
                    rank=alpha.comp_rank, # Storing raw rank normalized score for now
                    is_pegged=True, urgent=False,
                    tag=f"alpha_entry_{alpha.signal}"
                )
        
        # 2. Exit Logic
        elif alpha.is_exit and active_pos:
            is_long = active_pos.quantity > 0
            if (is_long and alpha.signal < 0) or (not is_long and alpha.signal > 0):
                intent = TradeIntent(
                    symbol=symbol, side='SELL' if is_long else 'BUY',
                    quantity=abs(active_pos.quantity),
                    snapshot=snap,
                    alpha=alpha.score,
                    rank=alpha.comp_rank,
                    is_pegged=False, urgent=True,
                    tag="alpha_exit"
                )

        if intent:
            try:
                self.signal_gate.mark_in_flight(symbol)
                self.intent_queue.put_nowait(intent)
                self.logger.debug(f"Queued intent for {symbol}: {intent.side}")
            except asyncio.QueueFull:
                self.logger.warning(f"Intent queue full! Dropping signal for {symbol}")
                self.signal_gate.clear_in_flight(symbol)

    async def _execution_worker(self):
        """
        Consumes TradeIntents and interfaces with ExecutionEngine.
        Runs in the async loop.
        """
        self.logger.info("Starting Execution Worker...")
        while not self._stop_event.is_set():
            try:
                # Wait for next intent
                intent = await asyncio.wait_for(self.intent_queue.get(), timeout=1.0)
                
                # Process Execute
                try:
                    self.logger.info(f"Worker Processing {intent.side} {intent.symbol}...")
                    
                    # Final feedback loop check before hitting ExecutionEngine
                    # ExecutionEngine also has pre-trade risk checks
                    local_id = self.execution.place_order(
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=intent.quantity,
                        snapshot=intent.snapshot,
                        tag=intent.tag,
                        is_pegged=intent.is_pegged,
                        urgent=intent.urgent,
                        alpha=intent.alpha,
                        rank=intent.rank
                    )
                    
                    if local_id:
                        self.logger.info(f"Order Placed successfully: {local_id}")
                    else:
                        self.logger.warning(f"Order Placement REJECTED by ExecutionEngine for {intent.symbol}")
                
                finally:
                    # Always clear signal gate to allow next signal after cooldown
                    self.intent_queue.task_done()
                    self.signal_gate.clear_in_flight(intent.symbol)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in execution worker: {e}", exc_info=True)

    def _is_market_open_for_trading(self) -> bool:
        """Checks if current time is within trading session hours."""
        now = datetime.now(IST)
        
        # Check weekdays
        if now.weekday() >= 5: # Sat=5, Sun=6
            return False
            
        start_time = now.replace(hour=self.settings.market_open_hour, 
                                 minute=self.settings.market_open_minute, 
                                 second=0, microsecond=0)
        start_time += timedelta(minutes=self.settings.no_trade_start_minutes)
        
        end_time = now.replace(hour=self.settings.market_close_hour, 
                               minute=self.settings.market_close_minute, 
                               second=0, microsecond=0)
        end_time -= timedelta(minutes=self.settings.no_trade_end_minutes)
        
        return start_time <= now <= end_time

    async def _session_management_loop(self):
        """Handles market open/close events and auto-flattening."""
        self.logger.info("Starting Session Management Loop...")
        while not self._stop_event.is_set():
            try:
                now = datetime.now(IST)
                
                # Auto-Flatten Logic
                # Triggered at (Market Close - no_trade_end_minutes)
                flatten_time = now.replace(hour=self.settings.market_close_hour, 
                                           minute=self.settings.market_close_minute, 
                                           second=0, microsecond=0)
                flatten_time -= timedelta(minutes=self.settings.no_trade_end_minutes)
                
                # Only run if within a 1-minute window of the flatten time on a weekday
                if now >= flatten_time and now < flatten_time + timedelta(minutes=1) and now.weekday() < 5:
                    await self._flatten_all_positions()
                    # Wait so we don't trigger repeatedly in the same minute
                    await asyncio.sleep(65)
                
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session loop: {e}")
                await asyncio.sleep(10)

    async def _flatten_all_positions(self):
        """Force close all open positions."""
        self.logger.warning("AUTO-FLATTEN TRIGGERED. Closing all positions.")
        positions = self.state_manager.get_open_positions()
        if not positions:
            self.logger.info("No positions to flatten.")
            return

        for pos in positions:
            side = 'SELL' if pos.quantity > 0 else 'BUY'
            self.logger.info(f"Flattening {pos.symbol} ({pos.quantity})...")
            # Bypass gate and queue for force-close safety
            self.execution.place_order(
                symbol=pos.symbol,
                side=side,
                quantity=abs(pos.quantity),
                tag="auto_flatten",
                urgent=True,
                alpha=0.0,
                rank=50.0
            )

    async def _monitoring_loop(self):
        """System health and state monitoring."""
        while not self._stop_event.is_set():
            try:
                # Use public accessors
                feat_stats = self.feature_engine.get_stats()
                stats = {
                    "active_instruments": feat_stats.get('instruments_tracked', 0),
                    "open_positions": len(self.state_manager.get_open_positions()),
                    "pending_orders": len(self.state_manager.get_open_orders()),
                    "ticker_connected": self.ticker.is_connected if self.ticker else False,
                    "queue_size": self.intent_queue.qsize()
                }
                self.logger.info(f"Heartbeat: {stats}")
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def shutdown(self):
        """Graceful and idempotent shutdown sequence."""
        async with self._shutdown_lock:
            if self._is_shutting_down:
                return
            self._is_shutting_down = True
            
        self.logger.info("Initiating Graceful Shutdown...")
        self._stop_event.set()
        
        # 1. Stop Data Flow
        if self.ticker:
            self.ticker.stop()
            self.logger.info("Ticker service stopped.")
            
        # 2. Stop Execution Monitor
        self.execution.stop()
        self.logger.info("Execution engine monitor stopped.")

        # 2.5 Drain Intent Queue
        if self.intent_queue.qsize() > 0:
            self.logger.info(f"Draining {self.intent_queue.qsize()} in-flight intents...")
            try:
                await asyncio.wait_for(self.intent_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Queue drain timed out. Forced shutdown.")
        
        # 3. Terminate background Tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self.logger.info("Background loops terminated.")
        
        # 4. Final Logs
        try:
            pnl, mtm = self.state_manager.get_total_daily_pnl()
            self.logger.info(f"Final Session Metrics - realized: {pnl:.2f}, MTM: {mtm:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to fetch final metrics: {e}")
            
        self.logger.info("Trading Engine Orchestrator Shutdown Complete.")

def handle_exit_signals(engine: TradingEngine):
    """
    Register system signals for clean exit.
    Note: Signal handlers must be set in the main thread's event loop.
    For Windows compatibility, this should be called with care.
    """
    try:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.shutdown()))
    except (NotImplementedError, AttributeError):
        # Windows event loop (ProactorLoop) does not support signal handlers
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    engine = TradingEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        # Handled by shutdown trigger in run() finally block
        pass
