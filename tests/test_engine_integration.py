import unittest
import asyncio
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock kiteconnect before ANY imports that might use it
mock_kiteconnect = MagicMock()
sys.modules['kiteconnect'] = mock_kiteconnect
sys.modules['kiteconnect.exceptions'] = MagicMock()

# Mock config before ANY imports that might use it
mock_config = MagicMock()
mock_settings = MagicMock()
mock_settings.broker_api_key = "ABC"
mock_settings.broker_access_token = "123"
mock_settings.rank_interval_sec = 0.1
mock_settings.weight_rank = 0.4
mock_settings.weight_wobi = 0.3
mock_settings.weight_anchor = 0.2
mock_settings.weight_regime = 0.1
mock_settings.entry_alpha_threshold = 0.7
mock_settings.exit_alpha_threshold = 0.3
mock_settings.default_order_quantity = 10
mock_settings.order_placement_delay_ms = 0
mock_settings.execution_marketable_limit_buffer_bps = 50.0
mock_settings.pegged_max_spread_bps = 5.0
mock_settings.max_spread_bps = 20.0
mock_settings.pegged_ttl_sec = 15
mock_settings.pegged_max_modifications = 2
mock_settings.order_poll_interval_ms = 100
mock_settings.use_pegged_orders = True
mock_settings.alpha_poll_interval_ms = 100
mock_settings.order_ttl_sec = 1
mock_settings.market_open_hour = 9
mock_settings.market_open_minute = 15
mock_settings.market_close_hour = 15
mock_settings.market_close_minute = 30
mock_settings.no_trade_start_minutes = 0
mock_settings.no_trade_end_minutes = 0
mock_settings.risk_max_volatility_annualized = 0.5
mock_settings.max_single_order_value = 500000.0

mock_config.get_settings.return_value = mock_settings
sys.modules['config.config'] = mock_config

from core.engine import TradingEngine
from state.state_manager import StateManager, OrderStatus, OrderSide
from data.ticker_service import TickData, OrderBookDepth

class TestEngineIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Mocking KiteConnect
        self.mock_kite = MagicMock()
        self.mock_kite.orders.return_value = []
        
        # Mock StateManager
        self.mock_state = MagicMock(spec=StateManager)
        self.mock_state.get_total_daily_pnl.return_value = (0.0, 0.0)
        
        # Patch create_ticker_service
        with patch('core.engine.create_ticker_service') as mock_create:
            self.mock_ticker = MagicMock()
            self.mock_ticker.start_async = AsyncMock()
            self.mock_ticker.is_connected.return_value = True
            self.mock_ticker.tick_store = MagicMock()
            mock_create.return_value = self.mock_ticker
            
            # Patch StateManager inside engine
            with patch('core.engine.StateManager', return_value=self.mock_state):
                with patch('threading.Thread'): # Prevent execution monitor thread
                    self.engine = TradingEngine(kite=self.mock_kite)

    async def test_signal_to_execution_wiring(self):
        """Tests if an alpha signal correctly reaches the execution queue and is processed."""
        
        # 1. Setup Alpha Engine to generate a buy signal
        self.engine.alpha_engine.compute_alpha = MagicMock()
        mock_alpha = MagicMock()
        mock_alpha.is_entry = True
        mock_alpha.signal = 1 # BUY
        mock_alpha.score = 0.8
        self.engine.alpha_engine.compute_alpha.return_value = mock_alpha
        
        # 2. Mock state/risk to allow entry
        self.mock_state.get_open_positions.return_value = []
        self.mock_state.get_open_orders.return_value = []
        self.engine.risk_manager.approve_trade = MagicMock(return_value=True)
        # Mock session to be open
        self.engine._is_market_open_for_trading = MagicMock(return_value=True)
        
        # 3. Create a mock snapshot
        from features.microstructure_features import MicrostructureSnapshot
        snap = MicrostructureSnapshot(
            instrument_token=1, symbol="RELIANCE", timestamp=datetime.now(),
            wobi=0.5, wobi_ema=0.5, last_price=2500, bid_depth_value=1, ask_depth_value=1,
            spread_bps=1, spread_normalized=0, mid_price=2500, microprice=2500,
            vwap=2500, price_vs_vwap=0, volume=1, buy_volume=1, sell_volume=0,
            volume_imbalance=1, trade_intensity=1, realized_vol_annualized=0.1,
            spread_volatility=0.1, wobi_volatility=0.1, price_momentum=0.1,
            wobi_momentum=0.1, liquidity_score=100, urgency_score=100,
            wobi_stable=1, wobi_persistence=1, is_stable=True, spoofing_flag=False,
            spoofing_score=0, depth_change_rate=0, tick_count=1
        )
        # Mock feature engine to return this snap
        self.engine.feature_engine.get_snapshot_by_symbol = MagicMock(return_value=snap)
        
        # 4. Trigger signal evaluation
        with patch.object(self.engine.execution, 'place_order', return_value="LOCAL_123") as mock_place:
            await self.engine._evaluate_signal(snap, mock_alpha)
            
            # Verify it's in the queue
            self.assertEqual(self.engine.intent_queue.qsize(), 1)
            
            # Now run the worker to process the queue
            worker_task = asyncio.create_task(self.engine._execution_worker())
            await self.engine.intent_queue.join()
            worker_task.cancel()
            
            # 5. Verify wiring
            mock_place.assert_called_once()
            _, kwargs = mock_place.call_args
            self.assertEqual(kwargs['symbol'], "RELIANCE")
            self.assertEqual(kwargs['side'], "BUY")
            self.assertTrue(kwargs['is_pegged'])

    async def test_shutdown_sequence(self):
        """Tests if the engine stops all services correctly."""
        self.engine.ticker = self.mock_ticker
        
        await self.engine.shutdown()
        
        self.assertTrue(self.engine._stop_event.is_set())
        self.mock_ticker.stop.assert_called_once()
        
    async def test_on_tick_received_wiring(self):
        """Tests if ticks from ticker service reach the feature engine."""
        mock_tick = MagicMock()
        with patch.object(self.engine.feature_engine, 'update') as mock_update:
            self.engine._on_tick_received(mock_tick)
            mock_update.assert_called_once_with(mock_tick)

    async def test_signal_gate_cooldown(self):
        """Tests if the SignalGate correctly blocks repeated signals."""
        from features.microstructure_features import MicrostructureSnapshot
        snap = MagicMock(spec=MicrostructureSnapshot)
        snap.symbol = "RELIANCE"
        
        mock_alpha = MagicMock()
        mock_alpha.is_entry = True
        mock_alpha.signal = 1
        
        self.mock_state.get_open_positions.return_value = []
        self.mock_state.get_open_orders.return_value = []
        
        # 1. First signal should pass
        await self.engine._evaluate_signal(snap, mock_alpha)
        self.assertEqual(self.engine.intent_queue.qsize(), 1)
        
        # 2. Second signal should be blocked by Gate (cooldown)
        await self.engine._evaluate_signal(snap, mock_alpha)
        self.assertEqual(self.engine.intent_queue.qsize(), 1) # Still 1

if __name__ == '__main__':
    unittest.main()
