import unittest
from unittest.mock import MagicMock
import sys

# Mock config
mock_config = MagicMock()
mock_settings = MagicMock()
mock_settings.max_positions = 2
mock_settings.max_sector_exposure_pct = 50.0 # 0.5 when divided by 100
mock_settings.max_daily_loss = 1000.0
mock_settings.max_consecutive_losses = 2
mock_settings.max_spread_bps = 10.0
mock_settings.risk_max_volatility_annualized = 0.5
mock_settings.max_single_order_value = 500000.0
mock_settings.risk_min_liquidity_daily_avg = 1000.0
mock_config.get_settings.return_value = mock_settings
sys.modules['config.config'] = mock_config
sys.modules['config'] = MagicMock()

from risk.risk_manager import RiskManager
from features.microstructure_features import MicrostructureSnapshot
from state.state_manager import StateManager, RiskFlag, Position, PositionSide, Order, OrderSide
from datetime import datetime

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock(spec=StateManager)
        self.risk = RiskManager(self.mock_state)
        # Mock sector map
        self.risk._sector_map = {"SYM1": "SEC1", "SYM2": "SEC1", "SYM3": "SEC2"}
        # Default Mocks
        self.mock_state.get_total_daily_pnl.return_value = (0.0, 0.0)
        self.mock_state.get_open_positions.return_value = []
        self.mock_state.get_open_orders.return_value = []
        self.mock_state.get_recent_closed_positions.return_value = []
        self.mock_state.is_trading_halted.return_value = False

    def test_max_positions_with_pending_orders(self):
        # Limit 2
        # 1 Open Position
        pos1 = MagicMock(symbol="SYM1"); pos1.quantity=10; pos1.current_price=100
        self.mock_state.get_open_positions.return_value = [pos1]
        
        # 1 Pending Order (New Symbol)
        ord1 = MagicMock(symbol="SYM2"); ord1.quantity=10; ord1.price=100
        self.mock_state.get_open_orders.return_value = [ord1]
        
        # Total Active = 2
        
        # New trade SYM3 -> Reject
        approved = self.risk.approve_trade("SYM3", "BUY", 10)
        self.assertFalse(approved)
        
        # New trade SYM1 (Existing) -> Approve
        approved_ext = self.risk.approve_trade("SYM1", "BUY", 10)
        self.assertTrue(approved_ext)

    def test_sector_exposure_notional(self):
        # Limit 0.4 (40%)
        # SYM1 (SEC1) Position: 10 * 100 = 1000 Value
        pos1 = MagicMock(symbol="SYM1"); pos1.quantity=10; pos1.current_price=100
        self.mock_state.get_open_positions.return_value = [pos1]
        self.mock_state.get_open_orders.return_value = []
        
        # Total Portfolio = 1000. SEC1 = 1000 (100%).
        
        # Trade SYM2 (SEC1). Qty 10 @ 100 = 1000.
        # New Total 2000. SEC1 2000 (100%). -> REJECT
        snap = MagicMock(spec=MicrostructureSnapshot)
        snap.last_price = 100.0
        snap.realized_vol_annualized = 0.1
        snap.spread_bps = 5.0
        snap.bid_depth_value = 10000.0
        snap.ask_depth_value = 10000.0
        
        approved = self.risk.approve_trade("SYM2", "BUY", 10, snapshot=snap)
        self.assertFalse(approved) 
        
        # Trade SYM3 (SEC2). Qty 100 @ 100 = 10000.
        # New Total = 1000 (pos) + 10000 (trade) = 11000.
        # SEC2 = 10000. Exposure 90% -> REJECT (Limit 40%)
        
        # Trade SYM3 (SEC2). Qty 10 @ 100 = 1000.
        # New Total 2000. SEC2 1000 (50%). -> REJECT (Limit 40%)
        
        # Wait, if I hold 1000 in SEC1.
        # And I buy 20000 worth of SEC2.
        # Total 21000. SEC2 20000 (95%).
        # What if I buy a tiny amount?
        # Trade SYM3 (SEC2) Qty 1 @ 100 = 100.
        # Total 1100. SEC2 100 (9%). -> APPROVE
        
        approved_small = self.risk.approve_trade("SYM3", "BUY", 1, snapshot=snap)
        self.assertTrue(approved_small)

    def test_peak_drawdown_circuit_breaker(self):
        # T0: PnL 0. Peak 0.
        self.mock_state.get_total_daily_pnl.return_value = (0.0, 0.0)
        self.risk.check_circuit_breakers()
        self.assertEqual(self.risk._peak_daily_pnl, 0.0)
        
        # T1: PnL +2000. Peak -> 2000.
        self.mock_state.get_total_daily_pnl.return_value = (2000.0, 0.0)
        self.risk.check_circuit_breakers()
        self.assertEqual(self.risk._peak_daily_pnl, 2000.0)
        
    def test_liquidity_multiplier_guard(self):
        # Order 10 @ 100 = 1000 Value.
        # Required Depth = 5000.
        
        snap = MagicMock(spec=MicrostructureSnapshot)
        snap.last_price = 100.0
        snap.realized_vol_annualized = 0.1
        snap.spread_bps = 5.0
        snap.bid_depth_value = 10000.0
        snap.ask_depth_value = 10000.0
        # Order 10 @ 100 = 1000 Value.
        # Required Depth = 5000.
        
        snap = MagicMock(spec=MicrostructureSnapshot)
        snap.last_price = 100.0
        snap.realized_vol_annualized = 0.1
        snap.spread_bps = 5.0
        snap.bid_depth_value = 0.0
        # Case 1: Low Depth (4000)
        snap.ask_depth_value = 4000.0
        approved_fail = self.risk.approve_trade("SYM_TEST", "BUY", 10, snapshot=snap)
        self.assertFalse(approved_fail)
        
        # Case 2: Enough Depth (6000)
        snap.ask_depth_value = 6000.0
        approved_pass = self.risk.approve_trade("SYM_TEST", "BUY", 10, snapshot=snap)
        self.assertTrue(approved_pass)

    def test_volatility_guard(self):
        snap = MagicMock(spec=MicrostructureSnapshot)
        snap.last_price = 100.0
        snap.spread_bps = 5.0
        snap.bid_depth_value = 0.0
        snap.ask_depth_value = 100000.0
        
        # High Vol (0.6 > 0.5 Limit)
        snap.realized_vol_annualized = 0.6
        approved = self.risk.approve_trade("SYM_TEST", "BUY", 10, snapshot=snap)
        self.assertFalse(approved)

    def test_consecutive_losses_closed_pos(self):
        # Mock limit 3 (mocked in setUp to be 2 in config? No, config limit is 3 usually).
        # Let's assume Limit from config is 2 for this test
        self.risk.max_consecutive_losses = 2
        
        p1 = MagicMock(); p1.realized_pnl = -10
        p2 = MagicMock(); p2.realized_pnl = -10
        
        self.mock_state.get_recent_closed_positions.return_value = [p1, p2]
        
        approved = self.risk.check_circuit_breakers()
        self.assertFalse(approved)
        self.mock_state.set_risk_flag.assert_called_with(RiskFlag.MAX_CONSECUTIVE_LOSSES, unittest.mock.ANY)

if __name__ == '__main__':
    unittest.main()
