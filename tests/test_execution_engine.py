import unittest
from unittest.mock import MagicMock, patch
import sys
import time
from datetime import datetime, timezone

# Mock config.config before importing ExecutionEngine
mock_config = MagicMock()
sys.modules['config.config'] = mock_config

from execution.execution_engine import ExecutionEngine
from state.state_manager import StateManager, OrderSide, OrderStatus
from risk.risk_manager import RiskManager

class TestExecutionEngine(unittest.TestCase):
    def setUp(self):
        self.mock_kite = MagicMock()
        self.mock_kite.configure_mock(**{
            "ORDER_TYPE_LIMIT": "LIMIT",
            "ORDER_TYPE_MARKET": "MARKET",
            "VARIETY_REGULAR": "regular",
            "PRODUCT_MIS": "MIS",
            "TRANSACTION_TYPE_BUY": "BUY",
            "TRANSACTION_TYPE_SELL": "SELL",
            "EXCHANGE_NSE": "NSE",
            "VALIDITY_DAY": "DAY"
        })
        
        # Mocking kite.orders() for reconciliation
        self.mock_kite.orders.return_value = []
        
        self.mock_state = MagicMock(spec=StateManager)
        self.mock_risk = MagicMock(spec=RiskManager)
        
        # Setup mock settings returned by get_settings()
        self.mock_settings = MagicMock()
        self.mock_settings.order_placement_delay_ms = 0
        self.mock_settings.execution_marketable_limit_buffer_bps = 50.0
        self.mock_settings.execution_pegged_ttl_sec = 15
        self.mock_settings.execution_pegged_max_modifications = 2
        self.mock_settings.execution_order_poll_interval_ms = 100
        self.mock_settings.use_pegged_orders = True
        
        mock_config.get_settings.return_value = self.mock_settings

        # Default: Always approve risk
        self.mock_risk.approve_trade.return_value = True

        # We patch the monitor thread to not start during tests
        with patch('threading.Thread'):
            self.engine = ExecutionEngine(self.mock_kite, self.mock_state, self.mock_risk)

    def test_reconciliation_on_startup(self):
        # 1 Managed pegged order, 1 Unmanaged order
        self.mock_kite.orders.return_value = [
            {'order_id': 'B_123', 'status': 'OPEN'}, # Managed
            {'order_id': 'B_999', 'status': 'OPEN'}  # Unmanaged
        ]
        
        managed_order = MagicMock()
        managed_order.order_id = 'L_123'
        managed_order.broker_order_id = 'B_123'
        managed_order.symbol = 'RELIANCE'
        managed_order.side = OrderSide.BUY
        managed_order.quantity = 10
        managed_order.filled_quantity = 0
        managed_order.price = 2500.0
        managed_order.tags = {"type": "pegged", "mods": 0}
        
        self.mock_state.get_open_orders.return_value = [managed_order]
        
        # Re-run reconciliation
        self.engine.reconcile_state()
        
        # B_123 should be in active tracking
        self.assertIn('L_123', self.engine._active_pegged_orders)
        # B_999 should be cancelled
        self.mock_kite.cancel_order.assert_called_with(variety="regular", order_id="B_999")
        
    def test_risk_rejection_at_execution(self):
        # Risk Manager rejects
        self.mock_risk.approve_trade.return_value = False
        
        order_id = self.engine.place_order(
            symbol="RELIANCE",
            side='BUY',
            quantity=10,
            price=2500.0
        )
        
        self.assertIsNone(order_id)
        self.mock_kite.place_order.assert_not_called()

    def test_place_marketable_limit_success(self):
        self.mock_kite.place_order.return_value = "BROKER_123"
        
        order_id = self.engine.place_order(
            symbol="RELIANCE",
            side='BUY',
            quantity=10,
            price=2500.0
        )
        
        self.assertIsNotNone(order_id)
        self.mock_kite.place_order.assert_called_once()
        self.mock_state.update_order_status.assert_called_with(
            order_id, OrderStatus.OPEN, broker_order_id="BROKER_123"
        )

    def test_pegged_tracking_with_partial_fill(self):
        self.mock_kite.quote.return_value = {
            "NSE:RELIANCE": {
                "depth": {"buy": [{"price": 2495.0}], "sell": [{"price": 2505.0}]}
            }
        }
        self.mock_kite.place_order.return_value = "B_PEG_1"
        
        order_id = self.engine.place_order(
            symbol="RELIANCE",
            side='BUY',
            quantity=10,
            is_pegged=True
        )
        
        # Initial check
        all_orders = {"B_PEG_1": {"order_id": "B_PEG_1", "status": "OPEN", "filled_quantity": 4}}
        quotes = {"NSE:RELIANCE": {"depth": {"buy": [{"price": 2495.0}]}}}
        
        self.engine._check_and_modify_pegged(order_id, all_orders, quotes)
        
        # Verify tracking updated correctly
        self.assertEqual(self.engine._active_pegged_orders[order_id]["remaining_qty"], 6)
        self.mock_state.update_order_status.assert_called_with(
            order_id, OrderStatus.PARTIAL_FILL, filled_quantity=4
        )

    def test_safe_fallback_logic(self):
        # Setup active pegged order with 2 mods (hits limit)
        local_id = "L_FB_1"
        self.engine._active_pegged_orders[local_id] = {
            "broker_id": "B_FB_1",
            "symbol": "RELIANCE",
            "side": "BUY",
            "original_qty": 10,
            "remaining_qty": 10,
            "price": 2495.0,
            "start_time": time.time(),
            "modifications": 2, # Limit reached
            "tag": "test"
        }
        
        # Mock broker cycle for cancel and recompute
        # 1. Check status (trigger fallback)
        all_orders = {"B_FB_1": {"order_id": "B_FB_1", "status": "OPEN", "filled_quantity": 2}}
        quotes = {"NSE:RELIANCE": {"depth": {"buy": [{"price": 2497.0}]}}}
        
        # 2. cancel and wait mock
        self.mock_kite.order_history.return_value = [
            {"status": "CANCELLED", "quantity": 10, "filled_quantity": 2}
        ]
        
        self.engine._check_and_modify_pegged(local_id, all_orders, quotes)
        
        # broker.cancel called
        self.mock_kite.cancel_order.assert_called_with("regular", "B_FB_1")
        
        # New marketable limit placed for REMAINING (10 - 2 = 8)
        # We look for a place_order call with quantity=8
        found_fallback = False
        for call in self.mock_kite.place_order.call_args_list:
            if call.kwargs.get('quantity') == 8:
                found_fallback = True
        self.assertTrue(found_fallback, "Fallback order for remaining quantity not placed.")

if __name__ == '__main__':
    unittest.main()
