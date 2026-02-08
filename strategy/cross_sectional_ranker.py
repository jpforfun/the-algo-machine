# strategy/cross_sectional_ranker.py
"""
Cross-sectional ranking module.

This module implements logic to rank instruments across the fixed NIFTY 50 universe
based on relative strength, microstructure signals, and other factors.

Key Features:
- Fixed Universe: Only ranks NIFTY 50 symbols.
- Z-Score Ranking: Ranks based on normalized Z-scores of combined signals.
- Configurable Weights: WOBI vs Momentum weights from config.
- Robust Persistence: Uses average percentile and population stdev.
- Thread-safe: Designed for single-process async/sync execution (atomic updates).
"""

import logging
import time
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set
from enum import Enum
import numpy as np

from config.config import get_settings
from features.microstructure_features import MicrostructureSnapshot

logger = logging.getLogger(__name__)

class RegimeState(Enum):
    """Market regime states."""
    TRENDING = "trending"
    CHOPPY = "choppy"
    HIGH_VOL = "high_vol"

class RegimeDetector:
    """
    Lightweight, deterministic regime detection.
    Uses config thresholds for lookback, volatility, range, and participation probability.
    """
    def __init__(self, settings):
        self.settings = settings
        self.lookback = settings.regime_trend_lookback_bars
        self.price_history = {}

    def update_price(self, symbol: str, price: float):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > self.lookback:
            self.price_history[symbol].pop(0)

    def detect_regime(self, symbol: str) -> RegimeState:
        if symbol not in self.price_history:
            return RegimeState.CHOPPY
        prices = self.price_history[symbol]
        if len(prices) < 5:
            return RegimeState.CHOPPY
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        if not returns:
            return RegimeState.CHOPPY
        volatility = np.std(returns) * 100
        price_range = (max(prices) - min(prices)) / np.mean(prices) * 100
        vol_threshold = getattr(self.settings, 'regime_volatility_high_threshold', 2.0)
        range_threshold = getattr(self.settings, 'regime_range_choppy_threshold', 1.0)
        vol_choppy_threshold = getattr(self.settings, 'regime_volatility_choppy_threshold', 1.5)
        if volatility > vol_threshold:
            return RegimeState.HIGH_VOL
        elif price_range < range_threshold and volatility < vol_choppy_threshold:
            return RegimeState.CHOPPY
        else:
            return RegimeState.TRENDING

    def get_regime_probability(self, symbol: str) -> float:
        regime = self.detect_regime(symbol)
        min_prob = self.settings.regime_min_participation_prob
        if regime == RegimeState.TRENDING:
            return 1.0
        elif regime == RegimeState.HIGH_VOL:
            return min_prob / 2
        else:
            return 0.0

@dataclass(kw_only=True)
class RankResult:
    """Ranking result for a single instrument."""
    instrument_token: int
    symbol: str
    rank: int  # 1-based rank (1 = best)
    percentile: float  # 0.0 to 100.0 (100 = best)
    raw_score: float  # Weighted combined score
    z_score: float  # Population Z-score of combined score
    active_signal: bool  # True if valid for ranking
    timestamp: datetime


class CrossSectionalRanker:
    """
    Ranks instruments based on multi-factor scoring.

    Factors:
    1. WOBI (Order Flow)
    2. Price Momentum
    """

    def __init__(self):
        self.settings = get_settings()
        self._ranks: Dict[int, RankResult] = {}
        self._last_update_time = 0.0
        # Configuration
        self.update_interval = self.settings.rank_interval_sec
        self.persistence_windows = self.settings.rank_persistence_windows
        self.wobi_weight = self.settings.rank_wobi_weight
        self.mom_weight = self.settings.rank_momentum_weight
        self.universe_symbols: Set[str] = self.settings.nifty50_symbols
        # History for persistence
        self._history: Dict[int, List[float]] = {}
        # Regime detector uses config thresholds
        self.regime_detector = RegimeDetector(self.settings)

    def get_rank_snapshot(self) -> Dict[int, 'RankResult']:
        """Public snapshot of current ranks (for engine safety)."""
        return dict(self._ranks)

    def update(self, snapshots: Dict[int, MicrostructureSnapshot]) -> None:
        now = time.monotonic()
        if now - self._last_update_time < self.update_interval:
            return
        self._compute_ranks(snapshots)
        self._last_update_time = now

    def _compute_ranks(self, snapshots: Dict[int, MicrostructureSnapshot]):
        if not snapshots:
            return
        scores = []
        valid_items = []
        for token, snapshot in snapshots.items():
            if snapshot.symbol not in self.universe_symbols:
                continue
            self.regime_detector.update_price(snapshot.symbol, snapshot.last_price)
            raw_score = (
                (snapshot.wobi_ema * self.wobi_weight) +
                (snapshot.price_momentum * self.mom_weight)
            )
            scores.append(raw_score)
            valid_items.append((token, snapshot.symbol, raw_score))
        if not scores or len(scores) < 2:
            return
        try:
            mean = statistics.mean(scores)
            stdev = statistics.pstdev(scores)
        except statistics.StatisticsError:
            mean = 0.0
            stdev = 0.0
        valid_items.sort(key=lambda x: x[2], reverse=True)
        timestamp = datetime.now()
        n = len(valid_items)
        for rank_0, (token, symbol, raw_score) in enumerate(valid_items):
            z_score = (raw_score - mean) / stdev if stdev > 0 else 0.0
            percentile = ((n - 1 - rank_0) / (n - 1)) * 100.0 if n > 1 else 50.0
            if token not in self._history:
                self._history[token] = []
            self._history[token].append(percentile)
            if len(self._history[token]) > self.persistence_windows + 5:
                self._history[token].pop(0)
            history = self._history.get(token, [])
            is_persistent = (
                len(history) >= self.persistence_windows and
                statistics.mean(history[-self.persistence_windows:]) >= self.settings.rank_top_percentile
            )
            active_signal = is_persistent
            result = RankResult(
                instrument_token=token,
                symbol=symbol,
                rank=rank_0 + 1,
                percentile=percentile,
                raw_score=raw_score,
                z_score=z_score,
                active_signal=active_signal,
                timestamp=timestamp
            )
            self._ranks[token] = result
        logger.debug(f"Updated ranks for {n} instruments")

    def get_rank(self, instrument_token: int) -> Optional[RankResult]:
        """Get ranking for a specific instrument."""
        return self._ranks.get(instrument_token)

    def get_top_candidates(self, n: int = 5) -> List[RankResult]:
        """Get best N ranked instruments (current snapshot)."""
        all_ranks = list(self._ranks.values())
        active = [r for r in all_ranks if r.active_signal]
        active.sort(key=lambda x: x.percentile, reverse=True)
        return active[:n]

    def get_bottom_candidates(self, n: int = 5) -> List[RankResult]:
        """Get worst N ranked instruments (shorts)."""
        all_ranks = list(self._ranks.values())
        active = [r for r in all_ranks if r.active_signal]
        active.sort(key=lambda x: x.percentile)
        return active[:n]

    def get_top_decile(self) -> List[RankResult]:
        """Get instruments in the top decile (top 10%)."""
        all_ranks = list(self._ranks.values())
        # Filter only active ranks
        active = [r for r in all_ranks if r.active_signal]
        active.sort(key=lambda x: x.percentile, reverse=True)
        return [r for r in active if r.percentile >= 90.0]

    def get_bottom_decile(self) -> List[RankResult]:
        """Get instruments in the bottom decile (bottom 10%)."""
        all_ranks = list(self._ranks.values())
        active = [r for r in all_ranks if r.active_signal]
        active.sort(key=lambda x: x.percentile)
        return [r for r in active if r.percentile <= 10.0]

    def get_persistent_top_candidates(self, n: int = 5, min_percentile: float = 80.0) -> List[RankResult]:
        """
        Get top candidates that have persisted in high percentiles.
        
        Logic: Average percentile over window >= min_percentile
        
        Args:
            n: Number of candidates to return
            min_percentile: Minimum average percentile required
        """
        candidates = []
        
        for token, result in self._ranks.items():
            history = self._history.get(token, [])
            if not history or len(history) < self.persistence_windows:
                continue
            
            # Use recent window
            recent = history[-self.persistence_windows:]
            avg_p = statistics.mean(recent)
            
            if avg_p >= min_percentile:
                candidates.append((result, avg_p))
        
        # Sort by average percentile descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [c[0] for c in candidates[:n]]
    
    def get_regime_state(self, symbol: str) -> RegimeState:
        """Get current regime state for symbol."""
        return self.regime_detector.detect_regime(symbol)
    
    def get_regime_prob(self, symbol: str) -> float:
        """Get regime probability for symbol."""
        return self.regime_detector.get_regime_probability(symbol)

# ================================================================
# TESTING
# ================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Ranker module available.")
