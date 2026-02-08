# risk/risk_manager.py
"""
Portfolio and Pre-Trade Risk Management.

This module acts as a gatekeeper for all trading activity.
It enforces:
1. Circuit breakers (Peak-to-Trough Drawdown, Consecutive Losses, Flags)
2. Portfolio constraints (Max Positions, Sector Exposure, Pending Orders)
3. Pre-trade guards (Spread, Volatility, Liquidity)

Method `approve_trade` returns True only if all checks pass.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
from threading import RLock

from config.config import get_settings
from state.state_manager import StateManager, RiskFlag, Position
from features.microstructure_features import MicrostructureSnapshot
from strategy.cross_sectional_ranker import RegimeState

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Central Risk Management Logic.
    """
    
    def __init__(self, state_manager: StateManager):
        self.settings = get_settings()
        self.state_manager = state_manager
        self._lock = RLock()
        
        # Load Limits
        self.max_positions = self.settings.max_positions
        self.max_sector_exposure = self.settings.max_sector_exposure_pct / 100.0
        # Peak-to-Trough Drawdown limit (calculated from capital %)
        self.max_drawdown_limit = (self.settings.total_capital * self.settings.max_drawdown_pct) / 100.0
        self.max_daily_loss = self.settings.max_daily_loss # Absolute currency limit (from 0)
        self.max_consecutive_losses = self.settings.max_consecutive_losses
        
        # Guard Thresholds
        self.max_spread_bps = self.settings.max_spread_bps
        self.max_vol_annualized = self.settings.risk_max_volatility_annualized
        self.min_liquidity_daily = self.settings.risk_min_liquidity_daily_avg
        
        # Position Sizing Parameters
        self.base_risk_pct = self.settings.base_risk_per_trade_pct / 100.0
        self.regime_multipliers = {
            RegimeState.TRENDING: self.settings.risk_multiplier_trending,
            RegimeState.CHOPPY: self.settings.risk_multiplier_choppy,
            RegimeState.HIGH_VOL: self.settings.risk_multiplier_high_vol,
        }
        
        # Internal State
        self._peak_daily_pnl = 0.0 # Track peak intraday PnL for trailing drawdown
        
        # Sector Map (Assuming NIFTY 50 top names)
        # TODO: Move to a proper master config or data layer
        self._sector_map = self._load_sector_map()

    def _load_sector_map(self) -> Dict[str, str]:
        """Load NIFTY 50 sector mapping."""
        return {
            "RELIANCE": "Oil & Gas", "ONGC": "Oil & Gas", "BPCL": "Oil & Gas",
            "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials", 
            "AXISBANK": "Financials", "KOTAKBANK": "Financials", "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials",
            "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
            "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
            "LT": "Construction",
            "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
            "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
            "TATAMOTORS": "Auto", "MARUTI": "Auto", "M&M": "Auto", "EICHERMOT": "Auto"
        }
    
    def calculate_position_size(self, 
                                symbol: str,
                                regime: RegimeState,
                                snapshot: MicrostructureSnapshot,
                                total_capital: float) -> Optional[int]:
        """
        Calculate volatility-aware position size.
        
        Formula:
            risk_fraction = base_risk * regime_multiplier
            position_size = (capital * risk_fraction) / stop_distance
        
        Args:
            symbol: Trading symbol
            regime: Current market regime
            snapshot: Market data for volatility estimation
            total_capital: Available capital
            
        Returns:
            Position size in shares, or None if cannot be computed safely
        """
        
        # Get regime-adjusted risk
        regime_multiplier = self.regime_multipliers.get(regime, 0.5)
        risk_fraction = self.base_risk_pct * regime_multiplier
        
        # Calculate stop distance
        # Use ATR proxy: realized_vol_annualized * price * 2.0 (2 standard deviations)
        if snapshot.realized_vol_annualized <= 0:
            logger.warning(f"Cannot size {symbol}: Invalid volatility")
            return None
        
        # Stop distance as 2x daily volatility in price units
        stop_distance = snapshot.last_price * snapshot.realized_vol_annualized * 2.0
        
        if stop_distance <= 0:
            logger.warning(f"Cannot size {symbol}: Invalid stop distance")
            return None
        
        # Calculate size
        position_value = total_capital * risk_fraction
        position_size = int(position_value / stop_distance)
        
        # Sanity checks
        if position_size <= 0:
            logger.warning(f"Cannot size {symbol}: Computed size <= 0")
            return None
        
        # Maximum single position value check
        max_position_value = self.settings.max_single_order_value
        actual_value = position_size * snapshot.last_price
        
        if actual_value > max_position_value:
            # Scale down to max value
            position_size = int(max_position_value / snapshot.last_price)
            logger.info(f"Position size scaled down for {symbol}: {actual_value:.0f} -> {max_position_value:.0f}")
        
        return position_size

    def check_circuit_breakers(self) -> bool:
        """
        Check global circuit breakers.
        Returns False if trading should stop.
        """
        # 0. Kill-switch Sovereignty (FIXED CRITICAL #1)
        active_flags = self.state_manager.get_active_risk_flags()
        if active_flags:
            if RiskFlag.SYSTEM_ERROR in active_flags:
                logger.critical("SYSTEM_ERROR active. Trading blocked.")
                return False

        with self._lock:
            # 1. Check Active Halted Flag
            if self.state_manager.is_trading_halted():
                return False
                
            # 2. Peak-to-Trough Drawdown
            # Get current PnL
            realized, unrealized = self.state_manager.get_total_daily_pnl()
            current_pnl = realized + unrealized
            
            # Update Peak
            if current_pnl > self._peak_daily_pnl:
                self._peak_daily_pnl = current_pnl
            
            # Calculate Drawdown
            drawdown = current_pnl - self._peak_daily_pnl
            
            # Check Limit (Drawdown is negative, max_drawdown_limit is positive limit)
            if drawdown < -self.max_drawdown_limit:
                logger.critical(f"Circuit Breaker: Peak-to-Trough Drawdown Limit Reached. "
                                f"Current: {drawdown:.2f}, Limit: -{self.max_drawdown_limit}")
                self.state_manager.set_risk_flag(RiskFlag.DRAWDOWN_BREACHED, {
                    "drawdown": drawdown,
                    "peak_pnl": self._peak_daily_pnl,
                    "current_pnl": current_pnl
                })
                return False
                
            # 3. Absolute Daily Loss (from 0)
            if current_pnl < -self.max_daily_loss:
                logger.critical(f"Circuit Breaker: Absolute Daily Loss Limit Reached. "
                                f"Current PnL: {current_pnl:.2f}, Limit: -{self.max_daily_loss}")
                self.state_manager.set_risk_flag(RiskFlag.MAX_DAILY_LOSS_BREACHED, {
                    "pnl": current_pnl,
                    "limit": self.max_daily_loss
                })
                return False
                
            # 3. Consecutive Losses (Position-based)
            closed_positions = self.state_manager.get_recent_closed_positions(limit=self.max_consecutive_losses)
            
            if len(closed_positions) >= self.max_consecutive_losses:
                # Check if all recent positions were losses
                consecutive_losses = 0
                for pos in closed_positions:
                    if pos.realized_pnl < 0:
                        consecutive_losses += 1
                    else:
                        break # Stop if we find a win
                
                if consecutive_losses >= self.max_consecutive_losses:
                    logger.critical(f"Circuit Breaker: Max Consecutive Losses ({self.max_consecutive_losses}) reached.")
                    self.state_manager.set_risk_flag(RiskFlag.MAX_CONSECUTIVE_LOSSES, {
                        "count": consecutive_losses
                    })
                    return False
                
            return True

    def approve_trade(self, 
                      symbol: str, 
                      side: str, 
                      qty: int, 
                      snapshot: Optional[MicrostructureSnapshot] = None,
                      is_urgent: bool = False) -> bool:
        """
        Main entry point for risk checks.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            qty: Quantity
            snapshot: Market data checks
            is_urgent: If True, allow trade regardless of portfolio limits (for exits/flattening)
            
        Returns:
            True if approved, False if rejected.
        """
        if qty <= 0:
            return False

        # 1. Global Circuit Breakers (Never bypass SYSTEM_ERROR or HALT)
        if not self.check_circuit_breakers():
            return False
            
        # 2. Market Data Heartbeat Guard (🛡️ Safety Improvement)
        # Check staleness of market data snapshot before any other pre-trade check.
        if snapshot:
            now_utc = datetime.now(timezone.utc)
            # Ensure snapshot.timestamp has timezone info or assume UTC
            ts = snapshot.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            
            age_sec = (now_utc - ts).total_seconds()
            threshold = self.settings.market_data_stale_threshold_sec
            
            if age_sec > threshold:
                logger.warning(f"Risk Veto {symbol}: Market data is STALE ({age_sec:.2f}s > {threshold}s). "
                               f"Heartbeat Guard triggered - Hard Veto.")
                return False

        # 3. Urgent Bypass: Allow reduction of risk regardless of size/constraints
        if is_urgent:
            return True
        
        # 🔴 CRITICAL: REGIME GATE (BLOCK ENTRIES IN CHOPPY)
        # This check happens early to prevent wasted processing
        # Note: We need regime state passed in - this will require caller update
        # For now, we check if snapshot indicates choppy conditions via volatility
        
        if side == 'BUY' and snapshot:
            # Proxy regime check via volatility
            # If volatility is very low AND spread is very tight, assume choppy
            # This is a simplified check - proper regime should come from ranker
            
            if (snapshot.realized_vol_annualized < 0.005 and  # < 0.5% volatility
                snapshot.spread_bps < 5.0):  # Very tight spread
                logger.info(f"Entry blocked for {symbol}: Detected CHOPPY conditions "
                           f"(vol={snapshot.realized_vol_annualized:.4f}, spread={snapshot.spread_bps:.1f}bps)")
                return False

        # 3. Max Order Value Guard (🛡️ Safety Improvement)
        if snapshot:
            order_value = qty * snapshot.last_price
            if order_value > self.settings.max_single_order_value:
                logger.warning(f"Risk Reject {symbol}: Order value ₹{order_value:,.2f} "
                               f"exceeds max limit ₹{self.settings.max_single_order_value:,.2f}")
                return False
            
        with self._lock:
            # Gather State
            open_positions = self.state_manager.get_open_positions()
            open_orders = self.state_manager.get_open_orders()
            
            # 2. Portfolio Constraints
            
            # A. Max Positions (Concurrent Open + Pending New)
            # We count distinct symbols in (Open Positions) UNION (Pending Entry Orders)
            active_symbols = {p.symbol for p in open_positions}
            
            # Add symbols from pending orders that would create NEW positions
            # (Simplification: Just count all symbols with pending orders if not already in positions)
            for order in open_orders:
                active_symbols.add(order.symbol)
            
            if len(active_symbols) >= self.max_positions:
                # If this symbol is NOT already active, reject
                if symbol not in active_symbols:
                    logger.warning(f"Risk Reject {symbol}: Max positions ({self.max_positions}) limit.")
                    return False
            
            # B. Sector Exposure (Notional Weighted)
            estimated_price = snapshot.last_price if snapshot else 0.0
            if estimated_price == 0.0 and symbol not in [p.symbol for p in open_positions]:
                 # Try to find price from existing usage or fail safe?
                 # If we can't price it, we can't check exposure properly.
                 # Assuming snapshot is mostly present for new entry.
                 pass
            
            sector = self._sector_map.get(symbol, "Unknown")
            if sector != "Unknown" and estimated_price > 0:
                # Calculate Total Portfolio Value and Sector Value
                total_value = 0.0
                sector_value = 0.0
                
                # Positions
                for pos in open_positions:
                    val = pos.quantity * pos.current_price # Mark to market
                    total_value += val
                    if self._sector_map.get(pos.symbol) == sector:
                        sector_value += val
                
                # Orders (Pending)
                for order in open_orders:
                    val = order.quantity * order.price # Limit price
                    total_value += val
                    if self._sector_map.get(order.symbol) == sector:
                        sector_value += val
                
                # Proposed Trade (FIXED CRITICAL #2)
                # direction = 1 for BUY, -1 for SELL (reduces exposure)
                direction = 1 if side == 'BUY' else -1
                trade_val = direction * qty * estimated_price
                
                new_total = max(0.0, total_value + trade_val)
                new_sector = max(0.0, sector_value + trade_val)
                
                if new_total > 0:
                    exposure = new_sector / new_total
                    if exposure > self.max_sector_exposure:
                        logger.warning(f"Risk Reject {symbol}: Max sector exposure ({sector} > {self.max_sector_exposure:.0%})")
                        return False

            # 3. Pre-Trade Guards (Microstructure)
            if snapshot:
                # A. Spread Guard
                # Use volatility-adjusted spread limit or strict cap? Strict cap for now.
                if snapshot.spread_bps > self.max_spread_bps:
                     logger.warning(f"Risk Reject {symbol}: Spread too wide ({snapshot.spread_bps:.1f} > {self.max_spread_bps})")
                     return False
                     
                # B. Volatility Guard
                # Use volatility-adjusted spread limit or strict cap? Strict cap for now.
                # Assuming snapshot.realized_vol_annualized is annualized.
                if snapshot.realized_vol_annualized > self.max_vol_annualized:
                    logger.warning(f"Risk Reject {symbol}: Volatility too high ({snapshot.realized_vol_annualized:.2f} > {self.max_vol_annualized})")
                    return False
                
                # C. Liquidity / Spoofing Guard
                # Rule: BidDepthValue < (OrderValue * Multiplier)
                # If asking to SELL, check BID depth. If BUY, check ASK depth.
                order_value = qty * snapshot.last_price
                required_depth = order_value * 5.0 # 5x coverage recommended
                
                available_depth = snapshot.ask_depth_value if side == 'BUY' else snapshot.bid_depth_value
                
                if available_depth < required_depth:
                     # Fallback: strict 0 check in case data is partial
                     if available_depth == 0:
                         logger.warning(f"Risk Reject {symbol}: Zero liquidity.")
                         return False
                     
                     # Soft warning for low liquidity? Or strict reject?
                     # User User Request: "bid_depth < 5 * order_qty -> reject"
                     # So Strict Reject.
                     logger.warning(f"Risk Reject {symbol}: Low liquidity. Depth {available_depth:.0f} < Required {required_depth:.0f}")
                     return False

        return True

    def reset_circuit_breaker(self):
        """Manual reset of risk flags."""
        with self._lock:
            self._peak_daily_pnl = 0.0 # Reset peak tracking
            
            # Clear flags in StateManager
            flags = [RiskFlag.DRAWDOWN_BREACHED, RiskFlag.MAX_CONSECUTIVE_LOSSES, RiskFlag.TRADING_HALTED]
            for f in flags:
                self.state_manager.clear_risk_flag(f)
            
            logger.info("Risk Manager flags and peak PnL manually reset.")
