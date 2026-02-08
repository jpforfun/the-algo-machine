import unittest
import sys
from unittest.mock import MagicMock
from datetime import datetime

# Mock config
mock_config = MagicMock()
mock_settings = MagicMock()
mock_settings.entry_alpha_threshold = 0.7
mock_settings.exit_alpha_threshold = 0.3
mock_settings.weight_rank = 0.4
mock_settings.weight_wobi = 0.3
mock_settings.weight_anchor = 0.2
mock_settings.weight_regime = 0.1
mock_config.get_settings.return_value = mock_settings
sys.modules['config.config'] = mock_config
sys.modules['config'] = MagicMock()

# Avoid importing real config/pydantic
from alpha.alpha_engine import AlphaEngine, AlphaScore
from features.microstructure_features import MicrostructureSnapshot
from strategy.cross_sectional_ranker import RankResult

class TestAlphaEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AlphaEngine()
        
    def create_mock_snapshot(self):
        return MicrostructureSnapshot(
            instrument_token=1, symbol="TEST", timestamp=datetime.now(),
            wobi=0.0, wobi_ema=0.0,
            bid_depth_value=1000, ask_depth_value=1000, spread_bps=10, spread_normalized=0,
            last_price=100, mid_price=100, microprice=100, vwap=100,
            price_vs_vwap=0.0, 
            volume=1000, buy_volume=500, sell_volume=500, volume_imbalance=0,
            trade_intensity=0, realized_vol_annualized=0, spread_volatility=0, wobi_volatility=0,
            price_momentum=0, wobi_momentum=0, liquidity_score=0, urgency_score=0,
            wobi_stable=0, wobi_persistence=0, is_stable=True, spoofing_flag=False,
            spoofing_score=0, depth_change_rate=0, tick_count=100
        )
        

    def create_mock_rank(self):
        return RankResult(
            instrument_token=1, symbol="TEST", rank=1, percentile=50.0,
            raw_score=0, z_score=0.0, active_signal=True, timestamp=datetime.now()
        )

    def test_neutral_score(self):
        """Test neutral inputs result in approx 0.5 score."""
        snap = self.create_mock_snapshot()
        rank = self.create_mock_rank()
        
        # WOBI 0 -> 0.5
        # Rank Z=0 -> 0.5
        # VWAP 0 -> Sigmoid(0) -> 0.5
        # Base Score -> 0.5 (assuming equal normalized weights)
        
        # Regime 0.5 -> Multiplier = 0.5 + 0.5*0.5 = 0.75
        # Final = 0.5 * 0.75 = 0.375
        
        res = self.engine.compute_alpha(snap, rank, regime_prob=0.5)
        
        # With current weights [Rank=0.4, WOBI=0.3, VWAP=0.2, Regime=0.1 (Removed from base)]
        # Re-normalized base weights: 
        # Total = 0.9.
        # w_r = 0.44, w_w = 0.33, w_v = 0.22
        # Base = 0.5 (all inputs 0.5)
        
        self.assertAlmostEqual(res.score, 0.375, delta=0.05)
        self.assertEqual(res.signal, 0) # 0.375 is Neutral (Between 0.3 and 0.7)
        
    def test_high_score(self):
        """Test strong bullish signals."""
        snap = self.create_mock_snapshot()
        snap.wobi_ema = 0.8 # Strong Buy -> ~0.9
        snap.price_vs_vwap = 2.0 # Max Bullish -> Tanh(2) ~0.96 -> val ~0.98
        
        rank = self.create_mock_rank()
        rank.z_score = 2.0 # Good Rank -> (2/6)+0.5 = 0.83
        
        # Regime: High confidence (1.0) -> Multiplier 1.0
        res = self.engine.compute_alpha(snap, rank, regime_prob=1.0)
        
        # Base approx weighted avg of (0.83, 0.9, 0.98) -> > 0.85
        # Multiplier 1.0
        # Final > 0.85 -> Entry
        
        self.assertGreater(res.score, 0.8)
        self.assertEqual(res.signal, 1)
        self.assertTrue(res.is_entry)

    def test_low_score(self):
        """Test strong bearish signals."""
        snap = self.create_mock_snapshot()
        snap.wobi_ema = -0.8 # Strong Sell -> 0.1
        snap.price_vs_vwap = -2.0 # Max Bearish -> 0.02
        
        rank = self.create_mock_rank()
        rank.z_score = -2.0 # Poor Rank -> 0.17
        
        res = self.engine.compute_alpha(snap, rank, regime_prob=1.0) 
        
        # Base < 0.2
        # Regime 1.0 (Valid Bear Signal needs good regime too usually, essentially we trust the signal)
        
        self.assertLess(res.score, 0.3)
        self.assertEqual(res.signal, -1)
        self.assertTrue(res.is_exit)

    def test_missing_rank_penalty(self):
        """Test missing rank penalizes score."""
        snap = self.create_mock_snapshot()
        # Neutral other signals
        
        res = self.engine.compute_alpha(snap, None, regime_prob=1.0)
        
        # Rank missing -> 0.4 DEFAULT
        # WOBI 0 -> 0.5
        # VWAP 0 -> 0.5
        # Base weighted avg will be < 0.5 due to rank penalty
        
        self.assertLess(res.score, 0.5)

if __name__ == '__main__':
    unittest.main()
