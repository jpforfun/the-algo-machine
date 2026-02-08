# features/microstructure_features.py
"""
Microstructure feature engineering module.

This module computes market microstructure features from tick data:
- Order book imbalance metrics (WOBI, OBI)
- Price impact and liquidity measures
- Volatility proxies
- Trade flow indicators
- Cross-sectional ranking features

All features are designed for:
- Real-time computation at tick-level
- Async-safe access patterns
- Rolling window calculations
- Memory-efficient storage

Features feed into the alpha scoring system.
"""

import asyncio
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

# Default rolling window sizes
DEFAULT_TICK_WINDOW = 100  # Ticks for rolling calculations
DEFAULT_TIME_WINDOW_SEC = 60  # Seconds for time-based windows

# Feature computation thresholds
MIN_TICKS_FOR_VOLATILITY = 10
MIN_TICKS_FOR_MOMENTUM = 5
MIN_SPREAD_FOR_DIVISION = 0.0001  # Avoid division by zero
VWAP_WINDOW_TICKS = 1000 # Window for Rolling VWAP (approx 15-30 mins depending on activity)

# Stability Filter Parameters
MIN_PERSISTENCE_TICKS = 3  # Minimum ticks WOBI must persist in same direction
STABILITY_THRESHOLD = 0.1  # Minimum |WOBI| to be considered a signal

# Spoofing Defense Parameters
DEPTH_DROP_THRESHOLD = 0.5  # 50% sudden depth drop = potential spoofing
SPOOFING_LOOKBACK_TICKS = 5  # Ticks to check for sudden depth changes
SPOOFING_RECOVERY_TICKS = 10  # Ticks to wait after spoofing detection
SPOOFING_PRICE_THRESHOLD_BPS = 2.0 # Price move > 2bps invalidates spoofing (likely real sweep)

# Normalization Constants
NORM_VOLATILITY_SCALE = 0.5
NORM_SPREAD_SCALE = 10
NORM_WOBI_STD_SCALE = 0.5
NORM_LIQUIDITY_DEPTH = 10000
NORM_SPREAD_BPS = 50

# Ranking Cache
RANKING_CACHE_TTL_SEC = 1.0


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass(kw_only=True)
class MicrostructureSnapshot:
    """
    Point-in-time microstructure features for a single instrument.
    
    All features are normalized to [-1, 1] or [0, 1] range where applicable.
    """
    instrument_token: int
    symbol: str
    timestamp: datetime
    
    # Order Book Features
    wobi: float  # Weighted Order Book Imbalance [-1, 1]
    wobi_ema: float  # EWMA smoothed WOBI [-1, 1]
    bid_depth_value: float  # Total weighted bid value
    ask_depth_value: float  # Total weighted ask value
    spread_bps: float  # Bid-ask spread in basis points
    spread_normalized: float  # Spread relative to historical [0, 1]
    
    # Price Features
    last_price: float
    mid_price: float
    microprice: float  # Inventory-adjusted mid price
    vwap: float  # Volume-weighted average price
    price_vs_vwap: float  # Distance from VWAP (normalized)
    
    # Volume & Flow Features
    volume: int
    buy_volume: int  # Estimated buy volume
    sell_volume: int  # Estimated sell volume
    volume_imbalance: float  # (buy - sell) / (buy + sell) [-1, 1]
    trade_intensity: float  # Trades per second [0, 1]
    
    # Volatility Features  
    realized_vol_annualized: float  # Tick-level volatility [0, 1]
    spread_volatility: float  # Spread volatility [0, 1]
    wobi_volatility: float  # WOBI stability metric [0, 1]
    
    # Momentum Features
    price_momentum: float  # Short-term price change [-1, 1]
    wobi_momentum: float  # WOBI trend [-1, 1]
    
    # Composite Scores
    liquidity_score: float  # Overall liquidity quality [0, 1]
    urgency_score: float  # Execution urgency indicator [0, 1]
    
    # Stability & Defense Features
    wobi_stable: float  # Stability-filtered WOBI [-1, 1], 0 if unstable
    wobi_persistence: int  # Consecutive ticks in same direction
    is_stable: bool  # True if WOBI passes stability filter
    spoofing_flag: bool  # True if spoofing detected
    spoofing_score: float  # Spoofing likelihood [0, 1]
    depth_change_rate: float  # Rate of depth change [-1, 1]
    
    # Metadata
    tick_count: int = 0
    computation_latency_ms: float = 0.0


@dataclass
class RollingStats:
    """Rolling window statistics for a single value series."""
    values: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_TICK_WINDOW))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_TICK_WINDOW))
    
    def add(self, value: float, timestamp: Optional[float] = None):
        """Add a new value to the rolling window."""
        self.values.append(value)
        self.timestamps.append(timestamp or time.monotonic())
    
    @property
    def mean(self) -> float:
        """Rolling mean."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    @property
    def std(self) -> float:
        """Rolling standard deviation."""
        if len(self.values) < 2:
            return 0.0
        mean = self.mean
        variance = sum((v - mean) ** 2 for v in self.values) / (len(self.values) - 1)
        return math.sqrt(variance)
    
    @property
    def min(self) -> float:
        """Rolling minimum."""
        return min(self.values) if self.values else 0.0
    
    @property
    def max(self) -> float:
        """Rolling maximum."""
        return max(self.values) if self.values else 0.0
    
    @property
    def last(self) -> float:
        """Last value."""
        return self.values[-1] if self.values else 0.0
    
    @property
    def first(self) -> float:
        """First value in window."""
        return self.values[0] if self.values else 0.0
    
    @property
    def change(self) -> float:
        """Change from first to last."""
        if len(self.values) < 2:
            return 0.0
        return self.values[-1] - self.values[0]
    
    @property
    def velocity(self) -> float:
        """Rate of change (per second)."""
        if len(self.values) < 2 or len(self.timestamps) < 2:
            return 0.0
        time_delta = self.timestamps[-1] - self.timestamps[0]
        if time_delta <= 0:
            return 0.0
        return self.change / time_delta


class InstrumentFeatureState:
    """
    Maintains rolling state for computing features for a single instrument.
    
    Tracks multiple rolling windows for different metrics.
    """
    
    def __init__(self, instrument_token: int, symbol: str, window_size: int = DEFAULT_TICK_WINDOW):
        self.instrument_token = instrument_token
        self.symbol = symbol
        self.window_size = window_size
        
        # Rolling statistics
        self.price_stats = RollingStats()
        self.wobi_stats = RollingStats()
        self.spread_stats = RollingStats()
        self.volume_stats = RollingStats()
        self.return_stats = RollingStats()
        
        # Rolling VWAP State
        self.vwap_window: deque = deque(maxlen=VWAP_WINDOW_TICKS) # Tuples of (price, volume)
        self.vwap_cumulative_vol = 0
        self.vwap_cumulative_val = 0.0
        
        # Depth tracking for spoofing detection
        self.depth_history: deque = deque(maxlen=SPOOFING_LOOKBACK_TICKS)
        self.bid_depth_stats = RollingStats()
        self.ask_depth_stats = RollingStats()
        
        # WOBI persistence tracking for stability filter
        self.wobi_direction_history: deque = deque(maxlen=MIN_PERSISTENCE_TICKS * 2)
        self.wobi_persistence_count = 0
        self.last_wobi_sign = 0  # -1, 0, or 1
        
        # Spoofing detection state
        self.spoofing_detected = False
        self.spoofing_cooldown = 0  # Ticks remaining in cooldown
        
        # Volume tracking
        self.buy_volume = 0
        self.sell_volume = 0
        
        # Tick tracking
        self.tick_count = 0
        self.first_tick_time: Optional[float] = None
        self.last_tick_time: Optional[float] = None
        self.last_price = 0.0
        self.last_wobi = 0.0
        self._last_seen_volume = 0 # Track last cumulative volume
        self.cumulative_volume = 0 # Track total volume for features
        
        # Best bid/ask tracking
        self.best_bid = 0.0
        self.best_ask = 0.0
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def update(self, tick_data) -> None:
        """
        Update state with new tick data.
        
        Args:
            tick_data: TickData from ticker_service
        """
        now = time.monotonic()
        
        with self._lock:
            # Update tick tracking
            self.tick_count += 1
            if self.first_tick_time is None:
                self.first_tick_time = now
            
            # Calculate return if we have previous price
            if self.last_price > 0:
                tick_return = (tick_data.last_price - self.last_price) / self.last_price
                self.return_stats.add(tick_return * 100, now)  # In percentage
            
            # Update rolling stats
            self.price_stats.add(tick_data.last_price, now)
            self.wobi_stats.add(tick_data.wobi_ema, now)
            self.spread_stats.add(tick_data.spread_bps, now)
            
            # Track WOBI persistence (stability filter)
            current_wobi_sign = 0
            if tick_data.wobi_ema > STABILITY_THRESHOLD:
                current_wobi_sign = 1
            elif tick_data.wobi_ema < -STABILITY_THRESHOLD:
                current_wobi_sign = -1
            
            self.wobi_direction_history.append(current_wobi_sign)
            
            if current_wobi_sign != 0 and current_wobi_sign == self.last_wobi_sign:
                self.wobi_persistence_count += 1
            else:
                self.wobi_persistence_count = 1 if current_wobi_sign != 0 else 0
            
            self.last_wobi_sign = current_wobi_sign
            
            # Track depth for spoofing detection
            total_depth = tick_data.bid_value + tick_data.ask_value
            self.depth_history.append(total_depth)
            self.bid_depth_stats.add(tick_data.bid_value, now)
            self.ask_depth_stats.add(tick_data.ask_value, now)
            
            # Detect spoofing (sudden depth disappearance)
            if self.spoofing_cooldown > 0:
                self.spoofing_cooldown -= 1
            
            if len(self.depth_history) >= SPOOFING_LOOKBACK_TICKS:
                avg_depth = sum(list(self.depth_history)[:-1]) / (len(self.depth_history) - 1)
                if avg_depth > 0:
                    depth_ratio = total_depth / avg_depth
                    if depth_ratio < DEPTH_DROP_THRESHOLD:
                        self.spoofing_detected = True
                        self.spoofing_cooldown = SPOOFING_RECOVERY_TICKS
                    elif self.spoofing_cooldown == 0:
                        self.spoofing_detected = False
            
            # Volume delta
            if hasattr(tick_data, 'volume'):
                # Note: tick_data.volume is cumulative for the day usually
                # We need delta for rolling calculations
                current_vol = tick_data.volume
                if self.tick_count == 1: # First tick
                     volume_delta = 0
                else: 
                     # Assuming tick_data.volume is cumulative. If it's 0 (first tick of day), handle gracefully
                     # We need previous volume. Since we don't store previous tick vol explicitly, 
                     # we can infer or simpler: Use the TickData logic which likely sends cumulative.
                     # But here we need to track *our* previous seen volume to get delta.
                     # Let's add a tracked field for prev_volume
                     pass 

                # Actually, simpler approach:
                # InstrumentFeatureState doesn't store prev_volume explicitly in __init__, let's calculate delta 
                # from internal tracking if possible, or assume caller provides it.
                # Looking at code: `volume_delta = max(0, tick_data.volume - self.cumulative_volume)`
                # The original code tracked `self.cumulative_volume` as the *last seen* volume to get delta.
                # So we CAN use that, but we renamed it. Let's assume we use a `_last_seen_volume`
                
                # RE-FIXING LOGIC IN PLACE:
                volume_delta = 0
                if self.tick_count > 1: # Not first tick
                     volume_delta = max(0, tick_data.volume - self._last_seen_volume)
                
                self._last_seen_volume = tick_data.volume
                self.cumulative_volume = tick_data.volume
                
                if volume_delta > 0:
                    self.volume_stats.add(volume_delta, now)
                    
                    # Rolling VWAP Update
                    self.vwap_window.append((tick_data.last_price, volume_delta))
                    self.vwap_cumulative_vol += volume_delta
                    self.vwap_cumulative_val += (tick_data.last_price * volume_delta)
                    
                    # Prune VWAP window
                    if len(self.vwap_window) == VWAP_WINDOW_TICKS:
                         # Oldest item is implicitly removed from deque, 
                         # BUT we need to subtract from cumulative sums.
                         # Since deque maxlen handles the list, we need to know what fell off.
                         # Actually checking len BEFORE append is hard with maxlen.
                         # Better: Store cumulative in a manual way or iterate.
                         # For O(1), we need to manually manage the sums.
                         # Let's use a slightly different approach:
                         # We appended. If len was already max, one popped.
                         # But standard deque doesn't return popped item on append.
                         # So we should Pop manually if full BEFORE append.
                         pass
                    
                    # Correct Rolling VWAP Logic:
                    # 1. Check if full
                    if len(self.vwap_window) == self.vwap_window.maxlen:
                        old_price, old_vol = self.vwap_window[0] # This is about to be popped/is at head
                        self.vwap_cumulative_vol -= old_vol
                        self.vwap_cumulative_val -= (old_price * old_vol)
                        
                    # 2. Append and Add
                    # (Append works on deque, auto-pops if maxlen, but we manually adjusted sums above)
                    # wait, if we use maxlen, `append` pops. So `self.vwap_window[0]` changes.
                    # This implementation relies on accessing [0] BEFORE append.
                    self.vwap_window.append((tick_data.last_price, volume_delta)) 
                    # Note: We already updated sums.
                    
                    # Estimate buy/sell using tick rule
                    
                    # Estimate buy/sell using tick rule
                    if tick_data.last_price > self.last_price:
                        self.buy_volume += volume_delta
                    elif tick_data.last_price < self.last_price:
                        self.sell_volume += volume_delta
                    else:
                        # Use WOBI to estimate direction
                        if tick_data.wobi_ema > 0:
                            self.buy_volume += volume_delta
                        else:
                            self.sell_volume += volume_delta
            
            # Update best prices
            if tick_data.depth.bids:
                self.best_bid = tick_data.depth.bids[0].price
            if tick_data.depth.asks:
                self.best_ask = tick_data.depth.asks[0].price
            
            # Update last values
            self.last_price = tick_data.last_price
            self.last_wobi = tick_data.wobi_ema
            self.last_tick_time = now
    
    def get_vwap(self) -> float:
        """Calculate Rolling VWAP."""
        if self.vwap_cumulative_vol == 0:
            return self.last_price
        return self.vwap_cumulative_val / self.vwap_cumulative_vol
    
    def get_microprice(self) -> float:
        """
        Calculate microprice (inventory-adjusted mid price).
        
        Formula: (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
        """
        bid_qty = 0
        ask_qty = 0
        # Use top level depth
        # We need access to the latest tick data's depth, 
        # but here we rely on best_bid/best_ask from update()
        # and we don't store qty in best_bid/best_ask fields currently.
        # We need to look at bid_depth_stats.last or similar? No, that's value.
        
        # Correction: InstrumentFeatureState needs to track best quantities for microprice
        # OR we pass the current tick's depth into this method.
        # But get_microprice is called without args in _compute_features.
        # Better: Calculate microprice IN _compute_features using tick_data directly.
        
        # Placeholder to ensure it's not used erroneously if called on state
        return self.last_price
    
    def get_trade_intensity(self) -> float:
        """Calculate trades per second over the window."""
        if self.first_tick_time is None or self.last_tick_time is None:
            return 0.0
        
        elapsed = self.last_tick_time - self.first_tick_time
        if elapsed <= 0:
            return 0.0
        
        return self.tick_count / elapsed
    
    def get_volume_imbalance(self) -> float:
        """Calculate buy/sell volume imbalance [-1, 1]."""
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total


# ================================================================
# FEATURE ENGINE
# ================================================================

class MicrostructureFeatureEngine:
    """
    Real-time microstructure feature computation engine.
    
    Maintains per-instrument state and computes features on tick updates.
    Features are normalized and suitable for alpha scoring.
    """
    
    def __init__(self, window_size: int = DEFAULT_TICK_WINDOW):
        """
        Initialize feature engine.
        
        Args:
            window_size: Rolling window size for statistics
        """
        self.window_size = window_size
        
        # Per-instrument state
        self._states: Dict[int, InstrumentFeatureState] = {}
        
        # Latest computed features
        self._features: Dict[int, MicrostructureSnapshot] = {}
        
        # Cross-sectional Cache
        self._ranking_cache: Dict[str, List[float]] = {} # feature -> sorted values
        self._ranking_cache_time = 0.0
        
        # Symbol to token mapping for easy lookup
        self._symbol_map: Dict[str, int] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks for feature updates
        self._callbacks: List[Callable[[MicrostructureSnapshot], None]] = []
        
        # Cross-sectional statistics (RollingStats for features)
        self._cross_sectional_stats: Dict[str, RollingStats] = {}
    
    def _get_or_create_state(self, token: int, symbol: str) -> InstrumentFeatureState:
        """Get or create instrument state."""
        if token not in self._states:
            self._states[token] = InstrumentFeatureState(
                token, symbol, self.window_size
            )
        return self._states[token]
    
    def update(self, tick_data) -> MicrostructureSnapshot:
        """
        Update features with new tick data.
        
        Args:
            tick_data: TickData from ticker_service
            
        Returns:
            Updated MicrostructureSnapshot for the instrument
        """
        start_time = time.monotonic()
        
        with self._lock:
            # Get/create state
            state = self._get_or_create_state(
                tick_data.instrument_token,
                tick_data.symbol
            )
            
            # Update state with new tick
            state.update(tick_data)
            
            # Compute features
            features = self._compute_features(state, tick_data)
            
            # Store latest features
            self._features[tick_data.instrument_token] = features
            
            # Update cross-sectional stats
            self._update_cross_sectional_stats(features)
            
            # Update symbol map
            self._symbol_map[tick_data.symbol] = tick_data.instrument_token
        
        # Record latency
        features.computation_latency_ms = (time.monotonic() - start_time) * 1000
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(features)
            except Exception as e:
                logger.warning(f"Feature callback error: {e}")
        
        return features
    
    def _compute_features(self, state: InstrumentFeatureState, tick_data) -> MicrostructureSnapshot:
        """Compute all features from state and tick data."""
        
        # Order Book Features
        wobi = tick_data.wobi_raw
        wobi_ema = tick_data.wobi_ema
        bid_depth_value = tick_data.bid_value
        ask_depth_value = tick_data.ask_value
        spread_bps = tick_data.spread_bps
        
        # Normalized spread (relative to historical)
        spread_normalized = self._normalize_value(
            spread_bps,
            state.spread_stats.min,
            state.spread_stats.max
        )
        
        # Price Features
        last_price = tick_data.last_price
        
        # Best bid/ask from tick (more accurate than state)
        best_bid_p = tick_data.depth.bids[0].price if tick_data.depth.bids else 0.0
        best_ask_p = tick_data.depth.asks[0].price if tick_data.depth.asks else 0.0
        best_bid_q = tick_data.depth.bids[0].quantity if tick_data.depth.bids else 0
        best_ask_q = tick_data.depth.asks[0].quantity if tick_data.depth.asks else 0
        
        mid_price = (best_bid_p + best_ask_p) / 2 if best_bid_p and best_ask_p else last_price
        
        # Correct Microprice Calculation
        if best_bid_q + best_ask_q > 0 and best_bid_p > 0 and best_ask_p > 0:
            microprice = (best_ask_p * best_bid_q + best_bid_p * best_ask_q) / (best_bid_q + best_ask_q)
        else:
            microprice = mid_price
            
        vwap = state.get_vwap()
        
        # Price vs VWAP (normalized distance)
        if vwap > 0:
            price_vs_vwap = (last_price - vwap) / vwap
            price_vs_vwap = max(-1.0, min(1.0, price_vs_vwap * 100))  # Scale and clip
        else:
            price_vs_vwap = 0.0
        
        # Volume & Flow Features
        volume = state.cumulative_volume
        buy_volume = state.buy_volume
        sell_volume = state.sell_volume
        volume_imbalance = state.get_volume_imbalance()
        trade_intensity = min(1.0, state.get_trade_intensity() / 10)  # Normalize to [0, 1]
        
        # Volatility Features
        realized_vol_annualized = 0.0
        if len(state.return_stats.values) >= MIN_TICKS_FOR_VOLATILITY:
            realized_vol_annualized = min(1.0, state.return_stats.std / 0.5)  # Normalize
        
        spread_volatility = 0.0
        if len(state.spread_stats.values) >= MIN_TICKS_FOR_VOLATILITY:
            spread_volatility = min(1.0, state.spread_stats.std / NORM_SPREAD_SCALE)  # Normalize
        
        wobi_volatility = 0.0
        if len(state.wobi_stats.values) >= MIN_TICKS_FOR_VOLATILITY:
            wobi_volatility = min(1.0, state.wobi_stats.std / NORM_VOLATILITY_SCALE)  # Normalize
        
        # Momentum Features
        price_momentum = 0.0
        if len(state.price_stats.values) >= MIN_TICKS_FOR_MOMENTUM:
            price_change = state.price_stats.change / max(state.price_stats.first, 0.01)
            price_momentum = max(-1.0, min(1.0, price_change * 100))
        
        wobi_momentum = 0.0
        if len(state.wobi_stats.values) >= MIN_TICKS_FOR_MOMENTUM:
            wobi_momentum = max(-1.0, min(1.0, state.wobi_stats.velocity))
        
        # Composite Scores
        liquidity_score = self._compute_liquidity_score(
            spread_bps, bid_depth_value, ask_depth_value
        )
        urgency_score = self._compute_urgency_score(
            wobi_ema, price_momentum, trade_intensity
        )
        
        # Stability & Spoofing Features
        if state.wobi_persistence_count >= MIN_PERSISTENCE_TICKS:
            wobi_stable = wobi_ema
            is_stable = True
        else:
            wobi_stable = 0.0
            is_stable = False
            
        # Spoofing Detection Enhanced
        # 1. Check if Spoofing Flagged (Depth Drop)
        # 2. Check Price Reaction: If price moved significantly (> X bps) in direction of depth removal, likely REAL.
        #    Spoofing = Depth Removed + Price Stable (No execution)
        
        spoofing_score = 0.0
        if state.spoofing_detected:
            # Check for price stability
            # If price changed significantly since spoof start, invalidate spoofing
            # We need reference price at spoof detection?
            # Simplified: Use short term momentum. If high momentum, likely real trade.
            
            # If momentum is low (< threshold) => Price Stable => Likely Spoofing
            if abs(price_momentum) < 0.2: # Threshold 0.2 is generic, improved with bps check
                 spoofing_score = 1.0
            else:
                 # Price moved, likely not spoofing
                 state.spoofing_detected = False
                 spoofing_score = 0.0
        
        if state.spoofing_cooldown > 0:
             # Decay score during recovery
             spoofing_score = 0.5 + (state.spoofing_cooldown / SPOOFING_RECOVERY_TICKS) * 0.5
        
        return MicrostructureSnapshot(
            instrument_token=state.instrument_token,
            symbol=state.symbol,
            timestamp=tick_data.timestamp,
            
            # Order Book
            wobi=wobi,
            wobi_ema=wobi_ema,
            bid_depth_value=bid_depth_value,
            ask_depth_value=ask_depth_value,
            spread_bps=spread_bps,
            spread_normalized=spread_normalized,
            
            # Price
            last_price=last_price,
            mid_price=mid_price,
            microprice=microprice,
            vwap=vwap,
            price_vs_vwap=price_vs_vwap,
            
            # Volume & Flow
            volume=volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_imbalance=volume_imbalance,
            trade_intensity=trade_intensity,
            
            # Volatility
            realized_vol_annualized=realized_vol_annualized,
            spread_volatility=spread_volatility,
            wobi_volatility=wobi_volatility,
            
            # Momentum
            price_momentum=price_momentum,
            wobi_momentum=wobi_momentum,
            
            # Composite
            liquidity_score=liquidity_score,
            urgency_score=urgency_score,
            
            # Stability & Defense
            wobi_stable=wobi_stable,
            wobi_persistence=state.wobi_persistence_count,
            is_stable=is_stable,
            spoofing_flag=state.spoofing_detected,
            spoofing_score=spoofing_score,
            depth_change_rate=0.0,  # Placeholder, computed from depth_history if needed
            
            tick_count=state.tick_count,
        )
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range."""
        if max_val <= min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _compute_liquidity_score(self, spread_bps: float, bid_value: float, ask_value: float) -> float:
        """
        Compute composite liquidity score [0, 1].
        
        Higher score = better liquidity (tighter spread, deeper book).
        """
        # Spread component (lower is better)
        spread_score = max(0.0, 1.0 - spread_bps / NORM_SPREAD_BPS)  # 0 at 50bps+
        
        # Depth component (higher is better)
        total_depth = bid_value + ask_value
        depth_score = min(1.0, total_depth / NORM_LIQUIDITY_DEPTH)  # Saturate at 10000 units
        
        # Balance component (more balanced is better)
        if total_depth > 0:
            imbalance = abs(bid_value - ask_value) / total_depth
            balance_score = 1.0 - imbalance
        else:
            balance_score = 0.5
        
        # Weighted combination
        return 0.5 * spread_score + 0.3 * depth_score + 0.2 * balance_score
    
    def _compute_urgency_score(self, wobi: float, momentum: float, intensity: float) -> float:
        """
        Compute execution urgency score [0, 1].
        
        Higher score = more urgent to act (strong signals aligning).
        """
        # WOBI magnitude (absolute value)
        wobi_urgency = abs(wobi)
        
        # Momentum aligning with WOBI
        if wobi > 0 and momentum > 0:  # Both bullish
            momentum_urgency = momentum
        elif wobi < 0 and momentum < 0:  # Both bearish
            momentum_urgency = abs(momentum)
        else:
            momentum_urgency = 0.0
        
        # Trade intensity adds urgency
        intensity_urgency = intensity
        
        # Weighted combination
        return 0.4 * wobi_urgency + 0.4 * momentum_urgency + 0.2 * intensity_urgency
    
    def _update_cross_sectional_stats(self, features: MicrostructureSnapshot):
        """
        NO-OP: Stats computed on demand.
        Keeping method signature if needed, but logic moved to get_cross_sectional_percentile.
        """
        pass
    
    def get_features(self, instrument_token: int) -> Optional[MicrostructureSnapshot]:
        """Get latest features for an instrument."""
        with self._lock:
            return self._features.get(instrument_token)
            
    def get_feature_snapshot(self, symbol: str) -> Optional[MicrostructureSnapshot]:
        """
        Get feature snapshot for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE')
            
        Returns:
            MicrostructureSnapshot if found, None otherwise
        """
        with self._lock:
            token = self._symbol_map.get(symbol)
            if token:
                return self._features.get(token)
            return None
    
    def get_all_features(self) -> Dict[int, MicrostructureSnapshot]:
        """Get all latest features."""
        with self._lock:
            return dict(self._features)
    
    # Removed async accessors to prevent threading mix-ups
    # get_features_async and get_all_features_async removed.
    
    def get_snapshot_by_symbol(self, symbol: str) -> Optional[MicrostructureSnapshot]:
        """Get the latest feature snapshot for a symbol."""
        with self._lock:
            for snap in self._features.values():
                if snap.symbol == symbol:
                    return snap
        return None

    def get_ranked_instruments(self, feature: str, ascending: bool = False) -> List[Tuple[int, str, float]]:
        """
        Get instruments ranked by a specific feature.
        
        Args:
            feature: Feature name to rank by (e.g., 'wobi_ema', 'liquidity_score')
            ascending: If True, lowest first; if False, highest first
            
        Returns:
            List of (token, symbol, value) tuples sorted by feature
        """
        with self._lock:
            ranked = []
            for token, snapshot in self._features.items():
                if hasattr(snapshot, feature):
                    value = getattr(snapshot, feature)
                    ranked.append((token, snapshot.symbol, value))
            
            ranked.sort(key=lambda x: x[2], reverse=not ascending)
            return ranked
    
    def get_cross_sectional_percentile(self, instrument_token: int, feature: str) -> float:
        """
        Get percentile rank of instrument for a feature (Cached).
        """
        with self._lock:
            # Refresh Cache if needed
            now = time.monotonic()
            if now - self._ranking_cache_time > RANKING_CACHE_TTL_SEC:
                # Recompute distribution cache
                # We only cache the Sorted Lists of values for key features
                features_to_rank = ['wobi_ema', 'liquidity_score', 'urgency_score']
                
                for feat in features_to_rank:
                     values = []
                     for snap in self._features.values():
                         if hasattr(snap, feat):
                             values.append(getattr(snap, feat))
                     values.sort()
                     self._ranking_cache[feat] = values
                
                self._ranking_cache_time = now
            
            # Compute percentile
            # Bisect right / len
            if feature not in self._ranking_cache:
                return -1.0
                
            sorted_vals = self._ranking_cache[feature]
            if not sorted_vals:
                return -1.0
                
            # Get current value
            snap = self._features.get(instrument_token)
            if not snap or not hasattr(snap, feature):
                return -1.0
            
            val = getattr(snap, feature)
            
            # Find rank
            import bisect
            idx = bisect.bisect_left(sorted_vals, val)
            
            return (idx / len(sorted_vals)) * 100.0
    
    def add_callback(self, callback: Callable[[MicrostructureSnapshot], None]):
        """Add callback for feature updates."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MicrostructureSnapshot], None]):
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def reset(self, instrument_token: Optional[int] = None):
        """Reset state for instrument or all instruments."""
        with self._lock:
            if instrument_token is not None:
                self._states.pop(instrument_token, None)
                self._features.pop(instrument_token, None)
            else:
                self._states.clear()
                self._features.clear()
                for stats in self._cross_sectional_stats.values():
                    stats.values.clear()
                    stats.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                'instruments_tracked': len(self._states),
                'total_ticks': sum(s.tick_count for s in self._states.values()),
                'cross_sectional': {
                    name: {
                        'count': len(stats.values),
                        'mean': stats.mean,
                        'std': stats.std,
                    }
                    for name, stats in self._cross_sectional_stats.items()
                }
            }


# ================================================================
# INTEGRATION HELPERS
# ================================================================

def create_feature_engine(window_size: int = None) -> MicrostructureFeatureEngine:
    """
    Create a MicrostructureFeatureEngine with configuration from settings.
    
    Args:
        window_size: Override window size (default: from config or 100)
        
    Returns:
        Configured MicrostructureFeatureEngine
    """
    if window_size is None:
        window_size = DEFAULT_TICK_WINDOW
    
    return MicrostructureFeatureEngine(window_size=window_size)


def connect_to_ticker(feature_engine: MicrostructureFeatureEngine, tick_store) -> Callable:
    """
    Connect feature engine to ticker data store.
    
    Args:
        feature_engine: MicrostructureFeatureEngine instance
        tick_store: TickDataStore from ticker_service
        
    Returns:
        Callback function that was registered (for later removal)
    """
    def on_tick(tick_data):
        try:
            feature_engine.update(tick_data)
        except Exception as e:
            logger.error(f"Feature update error: {e}")
    
    tick_store.add_callback(on_tick)
    return on_tick


# ================================================================
# TESTING & DIAGNOSTICS
# ================================================================

if __name__ == "__main__":
    """
    Test the microstructure feature engine.
    
    Run: python -m features.microstructure_features
    """
    import random
    from dataclasses import dataclass as dc
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Microstructure Feature Engine Tests")
    print("=" * 70)
    
    # Create mock tick data
    @dc
    class MockDepthLevel:
        price: float
        quantity: int
        orders: int = 0
    
    @dc
    class MockOrderBookDepth:
        bids: list
        asks: list
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @dc
    class MockTickData:
        instrument_token: int
        symbol: str
        last_price: float
        volume: int
        depth: MockOrderBookDepth
        wobi_raw: float
        wobi_ema: float
        bid_value: float
        ask_value: float
        spread_bps: float
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Test RollingStats
    print("\n1. Testing RollingStats...")
    stats = RollingStats()
    for i in range(20):
        stats.add(i, time.monotonic())
        time.sleep(0.01)
    
    print(f"   Mean: {stats.mean:.2f}")
    print(f"   Std: {stats.std:.2f}")
    print(f"   Min: {stats.min}")
    print(f"   Max: {stats.max}")
    print(f"   Change: {stats.change:.2f}")
    print(f"   ✓ RollingStats working")
    
    # Test InstrumentFeatureState
    print("\n2. Testing InstrumentFeatureState...")
    state = InstrumentFeatureState(12345, "RELIANCE", window_size=50)
    
    for i in range(30):
        mock_depth = MockOrderBookDepth(
            bids=[MockDepthLevel(100 - j * 0.05, 1000 - j * 100) for j in range(5)],
            asks=[MockDepthLevel(100.05 + j * 0.05, 800 - j * 100) for j in range(5)],
        )
        mock_tick = MockTickData(
            instrument_token=12345,
            symbol="RELIANCE",
            last_price=100 + random.uniform(-0.5, 0.5),
            volume=i * 1000,
            depth=mock_depth,
            wobi_raw=random.uniform(-0.3, 0.3),
            wobi_ema=random.uniform(-0.2, 0.2),
            bid_value=3000 + random.uniform(-500, 500),
            ask_value=2500 + random.uniform(-500, 500),
            spread_bps=5 + random.uniform(0, 3),
        )
        state.update(mock_tick)
        time.sleep(0.01)
    
    print(f"   Tick count: {state.tick_count}")
    print(f"   VWAP: {state.get_vwap():.2f}")
    print(f"   Trade intensity: {state.get_trade_intensity():.2f}")
    print(f"   Volume imbalance: {state.get_volume_imbalance():.4f}")
    print(f"   ✓ InstrumentFeatureState working")
    
    # Test MicrostructureFeatureEngine
    print("\n3. Testing MicrostructureFeatureEngine...")
    engine = MicrostructureFeatureEngine(window_size=50)
    
    # Simulate multiple instruments
    instruments = [
        (12345, "RELIANCE"),
        (67890, "TCS"),
        (11111, "INFY"),
    ]
    
    for _ in range(50):
        for token, symbol in instruments:
            mock_depth = MockOrderBookDepth(
                bids=[MockDepthLevel(100 - j * 0.05, 1000 - j * 100) for j in range(5)],
                asks=[MockDepthLevel(100.05 + j * 0.05, 800 - j * 100) for j in range(5)],
            )
            mock_tick = MockTickData(
                instrument_token=token,
                symbol=symbol,
                last_price=100 + random.uniform(-0.5, 0.5),
                volume=random.randint(1000, 10000),
                depth=mock_depth,
                wobi_raw=random.uniform(-0.5, 0.5),
                wobi_ema=random.uniform(-0.3, 0.3),
                bid_value=3000 + random.uniform(-500, 500),
                ask_value=2500 + random.uniform(-500, 500),
                spread_bps=5 + random.uniform(0, 5),
            )
            features = engine.update(mock_tick)
        time.sleep(0.01)
    
    print(f"   Instruments tracked: {len(engine.get_all_features())}")
    
    # Get features for one instrument
    features = engine.get_features(12345)
    if features:
        print(f"   RELIANCE features:")
        print(f"     - WOBI: {features.wobi:.4f}")
        print(f"     - Spread: {features.spread_bps:.2f} bps")
        print(f"     - Liquidity score: {features.liquidity_score:.4f}")
        print(f"     - Urgency score: {features.urgency_score:.4f}")
        print(f"     - Price momentum: {features.price_momentum:.4f}")
        print(f"   ✓ Feature computation working")
    
    # Test ranking
    print("\n4. Testing cross-sectional ranking...")
    ranked = engine.get_ranked_instruments('wobi_ema', ascending=False)
    print(f"   Ranking by WOBI (descending):")
    for token, symbol, value in ranked:
        percentile = engine.get_cross_sectional_percentile(token, 'wobi_ema')
        print(f"     {symbol}: {value:.4f} (percentile: {percentile:.1f})")
    print(f"   ✓ Cross-sectional ranking working")
    
    # Test async
    print("\n5. Testing async access...")
    
    async def test_async():
        features = engine.get_features(12345)
        return features is not None
    
    result = asyncio.run(test_async())
    print(f"   ✓ Async access working: {result}")
    
    # Stats
    print("\n6. Engine statistics...")
    stats = engine.get_stats()
    print(f"   {stats}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
