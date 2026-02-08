import unittest
import sys
from unittest.mock import MagicMock
from datetime import datetime

# Mock config before importing ranker
mock_config = MagicMock()
mock_settings = MagicMock()
mock_settings.rank_interval_sec = 1
mock_settings.rank_persistence_windows = 2
mock_settings.rank_wobi_weight = 0.5
mock_settings.rank_momentum_weight = 0.5
mock_settings.nifty50_symbols = {"SYM1", "SYM2", "SYM3"}
mock_config.get_settings.return_value = mock_settings
sys.modules['config.config'] = mock_config
sys.modules['config'] = MagicMock()

# Mock pydantic if needed (though avoiding import of config avoids pydantic import in config)
# But strategy.cross_sectional_ranker imports config.config
from strategy.cross_sectional_ranker import CrossSectionalRanker
from features.microstructure_features import MicrostructureSnapshot

class TestRanker(unittest.TestCase):
    def setUp(self):
        self.ranker = CrossSectionalRanker()
        
    def create_mock_snapshot(self, token, wobi, momentum):
        return MicrostructureSnapshot(
            instrument_token=token,
            symbol=f"SYM{token}",
            timestamp=datetime.now(),
            wobi=wobi,
            wobi_ema=wobi,
            bid_depth_value=1000.0,
            ask_depth_value=1000.0,
            spread_bps=10.0,
            spread_normalized=0.5,
            last_price=100.0,
            mid_price=100.0,
            microprice=100.0,
            vwap=100.0,
            price_vs_vwap=0.0,
            volume=1000.0,
            buy_volume=500.0,
            sell_volume=500.0,
            volume_imbalance=0.0,
            trade_intensity=0.5,
            realized_vol_annualized=0.1,
            spread_volatility=0.1,
            wobi_volatility=0.1,
            price_momentum=momentum,
            wobi_momentum=0.0,
            liquidity_score=0.8,
            urgency_score=0.5,
            wobi_stable=wobi,
            wobi_persistence=3,
            is_stable=True,
            spoofing_flag=False,
            spoofing_score=0.0,
            depth_change_rate=0.0,
            tick_count=100
        )

    def test_ranking(self):
        snapshots = {}
        # Token 1: High positive scores (Should be Rank 1)
        snapshots[1] = self.create_mock_snapshot(1, wobi=0.9, momentum=0.9)
        # Token 2: Neutral
        snapshots[2] = self.create_mock_snapshot(2, wobi=0.0, momentum=0.0)
        # Token 3: Negative (Should be last)
        snapshots[3] = self.create_mock_snapshot(3, wobi=-0.9, momentum=-0.9)
        
        self.ranker.update(snapshots)
        
        # Test Top Candidates
        top = self.ranker.get_top_candidates(3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0].instrument_token, 1)
        self.assertEqual(top[2].instrument_token, 3)
        
        # Test Percentiles
        r1 = self.ranker.get_rank(1)
        self.assertEqual(r1.percentile, 100.0)
        
        r2 = self.ranker.get_rank(2)
        self.assertEqual(r2.percentile, 50.0)
        
        r3 = self.ranker.get_rank(3)
        self.assertEqual(r3.percentile, 0.0)

    def test_ranking_persistence(self):
        snapshots = {}
        # Simulate 3 tokens
        # Token 1: Always good
        # Token 2: Good then bad
        # Token 3: Always bad
        
        # Update 1
        snapshots[1] = self.create_mock_snapshot(1, wobi=0.9, momentum=0.9)
        snapshots[2] = self.create_mock_snapshot(2, wobi=0.8, momentum=0.8)
        snapshots[3] = self.create_mock_snapshot(3, wobi=-0.9, momentum=-0.9)
        
        self.ranker.update(snapshots)
        
        # Force next update time
        self.ranker._last_update_time = 0
        
        # Update 2
        snapshots[1] = self.create_mock_snapshot(1, wobi=0.9, momentum=0.9) # Still good
        snapshots[2] = self.create_mock_snapshot(2, wobi=-0.8, momentum=-0.8) # Dropped
        snapshots[3] = self.create_mock_snapshot(3, wobi=-0.9, momentum=-0.9)
        
        self.ranker.update(snapshots)
        
        # Check Persistent Top Candidates (Window=2)
        # Token 1 should be there (Good -> Good)
        # Token 2 should NOT be there (Good -> Bad)
        persistent = self.ranker.get_persistent_top_candidates(n=1, min_percentile=50.0)
        
        self.assertEqual(len(persistent), 1)
        self.assertEqual(persistent[0].instrument_token, 1)
        
        # Check Deciles
        # With 3 items, percentiles are 100, 50, 0
        top_decile = self.ranker.get_top_decile() # >= 90
        # Rank 1 (Token 1) has 100.0 percentile
        self.assertEqual(len(top_decile), 1)
        self.assertEqual(top_decile[0].instrument_token, 1)

if __name__ == '__main__':
    unittest.main()
