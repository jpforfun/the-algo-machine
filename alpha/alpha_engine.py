# alpha/alpha_engine.py
"""
Alpha Scoring Engine.

This module is responsible for computing a continuous alpha score based on
multiple input signals. It enforces entry and exit thresholds to generate
discrete trading signals.

The Alpha Engine is STATELESS and pure. It relies on inputs from:
- Microstructure Features (WOBI, VWAP)
- Cross-Sectional Ranker
- Regime Detection

CRITICAL UPDATE: Now includes cost-aware signal veto to prevent negative expectancy trades.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from config.config import get_settings
from features.microstructure_features import MicrostructureSnapshot
from strategy.cross_sectional_ranker import RankResult

class AlphaHorizon(Enum):
    """Alpha signal horizon types."""
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"

@dataclass(kw_only=True)
class AlphaScore:
    """Result of alpha computation with cost-awareness."""
    raw_score: float  # Raw alpha before costs [0.0 to 1.0]
    effective_alpha: float  # After transaction costs (can be negative)
    score: float  # Final score used for thresholding [0.0 to 1.0]
    signal: int  # 1 (BUY), -1 (SELL), 0 (HOLD/NEUTRAL)
    is_entry: bool  # True if this score crosses entry threshold
    is_exit: bool  # True if this score crosses exit threshold
    cost_vetoed: bool  # True if signal was killed by cost veto
    horizon: AlphaHorizon  # Signal horizon
    
    # Component scores for debugging
    comp_rank: float
    comp_wobi: float
    comp_vwap: float
    comp_regime: float
    transaction_cost: float  # Applied cost in score units

class AlphaEngine:
    """
    Computes cost-aware alpha scores and generates signals.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Thresholds
        self.entry_threshold = self.settings.entry_alpha_threshold
        self.exit_threshold = self.settings.exit_alpha_threshold
        
        # Weights
        self.w_rank = self.settings.weight_rank
        self.w_wobi = self.settings.weight_wobi
        self.w_vwap = self.settings.weight_anchor
        self.w_regime = self.settings.weight_regime
        
        # Transaction costs
        self.transaction_cost_bps = self.settings.estimated_transaction_cost_bps
        self.transaction_cost_score = self.transaction_cost_bps / 10000.0  # Convert to [0,1] scale
        
        # Alpha horizon
        self.alpha_horizon = (AlphaHorizon.FIFTEEN_MIN 
                             if self.settings.alpha_horizon_minutes == 15 
                             else AlphaHorizon.FIVE_MIN)
        
        # Validation
        total_weight = self.w_rank + self.w_wobi + self.w_vwap + self.w_regime
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights if they don't sum to 1
            self.w_rank /= total_weight
            self.w_wobi /= total_weight
            self.w_vwap /= total_weight
            self.w_regime /= total_weight

    def compute_alpha(self, 
                      snapshot: MicrostructureSnapshot, 
                      rank: Optional[RankResult], 
                      regime_prob: float) -> AlphaScore:
        """
        Compute cost-aware alpha score from inputs.
        
        CRITICAL: Applies transaction cost veto before signal generation.
        
        Formula:
            Raw Score = w_rank * Norm(Rank_Z) + w_wobi * Norm(WOBI) + w_vwap * Norm_Sigmoid(VWAP)
            Base Score = Raw Score * (0.5 + 0.5 * RegimeProb)
            Effective Alpha = Base Score - Transaction_Cost
            Final Score = max(0, Effective Alpha)  # Clamped
            
        Args:
            snapshot: Microstructure features
            rank: Ranking result (optional)
            regime_prob: Probability of favorable regime (0.0 - 1.0)
            
        Returns:
            AlphaScore object with cost-adjusted score and veto status.
        """
        
        # 1. Rank Component [0, 1]
        # Use Z-Score if available for better resolution, else Percentile.
        # Fallback to 0.4 (slightly bearish/weak) if missing, to penalize uncertainty.
        val_rank = 0.4 
        if rank:
            # Prefer Z-Score clamped to +/- 3
            # Z=0 -> 0.5, Z=3 -> 1.0, Z=-3 -> 0.0
            z_clamped = max(-3.0, min(3.0, rank.z_score))
            val_rank = (z_clamped / 6.0) + 0.5
            
        # 2. WOBI Component [0, 1]
        # WOBI is [-1, 1]. Map to [0, 1].
        val_wobi = (snapshot.wobi_ema + 1.0) / 2.0
        val_wobi = max(0.0, min(1.0, val_wobi))
        
        # 3. VWAP Component [0, 1] using Sigmoid (Tanh)
        # Tanh(x) maps (-inf, inf) to (-1, 1).
        # We want Z=0 -> 0.5.
        # Scale: 1% deviation (approx 100bps) should be significant.
        # snapshot.price_vs_vwap is percentage (e.g., 1.0 = 1%).
        # tanh(1.0) ~= 0.76. This seems reasonable.
        # val_vwap = 0.5 + 0.5 * tanh(price_vs_vwap)
        val_vwap = 0.5 + 0.5 * math.tanh(snapshot.price_vs_vwap)
        
        # Combine Components (Raw Score)
        # Re-normalize weights for just these 3 components
        w_total = self.w_rank + self.w_wobi + self.w_vwap
        w_r = self.w_rank / w_total
        w_w = self.w_wobi / w_total
        w_v = self.w_vwap / w_total
        
        raw_score = (
            w_r * val_rank +
            w_w * val_wobi +
            w_v * val_vwap
        )
        
        # 4. Regime Gating
        # Multiplier approach: Suppress signals in bad regimes.
        # If regime=1.0 -> Score = Base
        # If regime=0.0 -> Score = Base * 0.5 (Dampened)
        regime_multiplier = 0.5 + (0.5 * max(0.0, min(1.0, regime_prob)))
        
        base_score = raw_score * regime_multiplier
        base_score = max(0.0, min(1.0, base_score))
        
        # 🔴 CRITICAL: COST-AWARE VETO
        # Subtract transaction cost from base score
        effective_alpha = base_score - self.transaction_cost_score
        
        # Hard veto: If effective alpha is non-positive, KILL the signal
        cost_vetoed = effective_alpha <= 0.0
        
        # Final score is clamped effective alpha (for threshold checks)
        final_score = max(0.0, effective_alpha)
        
        # Signal Logic (only if NOT vetoed)
        signal = 0
        is_entry = False
        is_exit = False
        
        if not cost_vetoed:
            is_entry = final_score >= self.entry_threshold
            is_exit = final_score <= self.exit_threshold
            
            if is_entry:
                signal = 1
            elif is_exit:
                signal = -1
            
        return AlphaScore(
            raw_score=raw_score,
            effective_alpha=effective_alpha,
            score=final_score,
            signal=signal,
            is_entry=is_entry,
            is_exit=is_exit,
            cost_vetoed=cost_vetoed,
            horizon=self.alpha_horizon,
            comp_rank=val_rank,
            comp_wobi=val_wobi,
            comp_vwap=val_vwap,
            comp_regime=regime_multiplier,
            transaction_cost=self.transaction_cost_score
        )

# Simple Test Block
if __name__ == "__main__":
    import datetime
    
    # Mock Objects
    snap = MicrostructureSnapshot(
        instrument_token=1, symbol="TEST", timestamp=datetime.datetime.now(),
        wobi=0.5, wobi_ema=0.5, # Strong Buy
        bid_depth_value=1, ask_depth_value=1, spread_bps=1, spread_normalized=0,
        last_price=100, mid_price=100, microprice=100, vwap=99,
        price_vs_vwap=1.0, # +1% vs VWAP (Bullish)
        volume=100, buy_volume=50, sell_volume=50, volume_imbalance=0,
        trade_intensity=0, realized_vol_annualized=0.0, spread_volatility=0, 
        wobi_volatility=0, price_momentum=0, wobi_momentum=0,
        liquidity_score=0, urgency_score=0, wobi_stable=0, wobi_persistence=0,
        is_stable=True, spoofing_flag=False, spoofing_score=0, depth_change_rate=0,
        tick_count=10
    )
    
    rank = RankResult(
        instrument_token=1, symbol="TEST", rank=1, percentile=90.0, 
        raw_score=1.0, z_score=2.0, active_signal=True, timestamp=datetime.datetime.now()
    )
    
    engine = AlphaEngine()
    
    # Test High Score
    res = engine.compute_alpha(snap, rank, regime_prob=0.8)
    print(f"High Score Test: raw={res.raw_score:.4f} effective={res.effective_alpha:.4f} final={res.score:.4f} Signal: {res.signal} Vetoed: {res.cost_vetoed}")
    print(f"Components: Rank={res.comp_rank:.2f} WOBI={res.comp_wobi:.2f} VWAP={res.comp_vwap:.2f} Regime={res.comp_regime:.2f}")
    print(f"Horizon: {res.horizon.value}, TxnCost: {res.transaction_cost:.4f}")

    # Test Low Score (should be vetoed)
    snap.wobi_ema = -0.5
    snap.price_vs_vwap = -1.0
    rank.percentile = 10.0
    rank.z_score = -2.0
    res_low = engine.compute_alpha(snap, rank, regime_prob=0.2)
    print(f"\nLow Score Test: raw={res_low.raw_score:.4f} effective={res_low.effective_alpha:.4f} final={res_low.score:.4f} Signal: {res_low.signal} Vetoed: {res_low.cost_vetoed}")
