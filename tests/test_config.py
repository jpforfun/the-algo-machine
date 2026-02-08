# tests/test_config.py
"""
Comprehensive test suite for configuration validation.

Tests all fixes and improvements:
1. Depth weights matching depth_levels
2. Deferred singleton pattern
3. Spread consistency (both in bps)
4. Alpha threshold ordering
5. Immutability (frozen config)
"""

import pytest
from pathlib import Path
from datetime import datetime, time
import pytz
from pydantic import ValidationError


def test_depth_weights_match_depth_levels():
    """Test that depth_weights validation respects depth_levels setting."""
    from config.config import Settings
    
    # Valid: 5 weights for 5 levels (default)
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        depth_levels=5,
        depth_weights=[1.0, 0.8, 0.6, 0.4, 0.2]
    )
    assert len(config.depth_weights) == config.depth_levels
    
    # Valid: 3 weights for 3 levels
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        depth_levels=3,
        depth_weights=[1.0, 0.7, 0.4]
    )
    assert len(config.depth_weights) == 3
    
    # Invalid: 5 weights for 3 levels
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            depth_levels=3,
            depth_weights=[1.0, 0.8, 0.6, 0.4, 0.2]
        )
    assert "depth_weights length" in str(exc.value)


def test_depth_weights_descending_order():
    """Test that depth weights must be in descending order."""
    from config.config import Settings
    
    # Invalid: weights not in descending order
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            depth_levels=5,
            depth_weights=[1.0, 0.6, 0.8, 0.4, 0.2]  # 0.6 < 0.8 violates order
        )
    assert "descending order" in str(exc.value)


def test_deferred_singleton_pattern():
    """Test that get_settings() uses deferred loading."""
    from config.config import get_settings, reset_settings, _settings
    import os
    from unittest.mock import patch
    
    # Mock environment variables
    with patch.dict(os.environ, {
        'TRADING_BROKER_API_KEY': 'test_key', 
        'TRADING_BROKER_ACCESS_TOKEN': 'test_token'
    }):
        # Initially None
        reset_settings()
        
        # First call loads settings
        settings1 = get_settings()
        assert settings1 is not None
        assert settings1.broker_api_key == 'test_key'
        
        # Second call returns same instance
        settings2 = get_settings()
        assert settings1 is settings2
        
        # Reset and verify new instance created
        reset_settings()
        settings3 = get_settings()
        assert settings3 is not settings1


def test_spread_consistency_both_bps():
    """Test that both spread parameters use basis points and are consistent."""
    from config.config import Settings
    
    # Valid: pegged spread < general spread
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        max_spread_bps=20,
        pegged_max_spread_bps=5
    )
    assert config.pegged_max_spread_bps < config.max_spread_bps
    
    # Invalid: pegged spread >= general spread
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            max_spread_bps=20,
            pegged_max_spread_bps=25
        )
    assert "pegged_max_spread_bps" in str(exc.value)


def test_alpha_threshold_ordering():
    """Test that alpha thresholds satisfy exit < entry < urgent."""
    from config.config import Settings
    
    # Valid ordering
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        exit_alpha_threshold=0.30,
        entry_alpha_threshold=0.65,
        urgent_alpha_threshold=0.80
    )
    assert config.exit_alpha_threshold < config.entry_alpha_threshold < config.urgent_alpha_threshold
    
    # Invalid: exit >= entry
    # Note: Must use values within field ranges
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            exit_alpha_threshold=0.5,
            entry_alpha_threshold=0.5, # 0.5 >= 0.5, violates logical ordering
            urgent_alpha_threshold=0.80
        )
    assert "exit < entry < urgent" in str(exc.value)
    
    # Invalid: entry >= urgent
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            exit_alpha_threshold=0.30,
            entry_alpha_threshold=0.80,
            urgent_alpha_threshold=0.80 # 0.80 >= 0.80
        )
    assert "exit < entry < urgent" in str(exc.value)


def test_alpha_weights_sum_to_one():
    """Test that alpha component weights sum to 1.0."""
    from config.config import Settings
    
    # Valid: sums to 1.0
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        weight_regime=0.35,
        weight_rank=0.25,
        weight_wobi=0.20,
        weight_anchor=0.20
    )
    total = config.weight_regime + config.weight_rank + config.weight_wobi + config.weight_anchor
    assert 0.99 <= total <= 1.01
    
    # Invalid: doesn't sum to 1.0
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            weight_regime=0.40,
            weight_rank=0.30,
            weight_wobi=0.20,
            weight_anchor=0.20  # Sum = 1.10
        )
    assert "must sum to 1.0" in str(exc.value)


def test_config_immutability():
    """Test that configuration is frozen and cannot be modified."""
    from config.config import Settings
    
    config = Settings(
        broker_api_key="test",
        broker_access_token="test"
    )
    
    # Attempt to modify should raise error
    with pytest.raises(ValidationError):
        config.max_positions = 10


def test_market_hours_validation():
    """Test market hours logical consistency."""
    from config.config import Settings
    
    # Valid market hours
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        market_open_hour=9,
        market_open_minute=15,
        market_close_hour=15,
        market_close_minute=30
    )
    assert config.market_open_hour < config.market_close_hour
    
    # Invalid: close before open
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            market_open_hour=15,
            market_open_minute=30,
            market_close_hour=9,
            market_close_minute=15
        )
    assert "close time must be after" in str(exc.value)


def test_no_trade_windows_validation():
    """Test that no-trade windows don't exceed market hours."""
    from config.config import Settings
    
    # Valid: buffers within market hours
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        market_open_hour=9,
        market_open_minute=15,
        market_close_hour=15,
        market_close_minute=30,
        no_trade_start_minutes=15,
        no_trade_end_minutes=15
    )
    
    # Invalid: buffers exceed market hours
    # We set a short market window to test this, since no_trade_minutes are capped at 60
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            market_open_hour=9,
            market_open_minute=0,
            market_close_hour=10, # 1 hour window
            market_close_minute=0,
            no_trade_start_minutes=40,
            no_trade_end_minutes=30  # Total 70 mins > 60 mins window
        )
    assert "exceed total market hours" in str(exc.value)


def test_risk_limits_consistency():
    """Test risk limit logical consistency."""
    from config.config import Settings
    
    # Valid: sector positions <= max positions
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        max_positions=5,
        max_sector_positions=2
    )
    assert config.max_sector_positions <= config.max_positions
    
    # Invalid: sector positions > max positions
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            max_positions=5,
            max_sector_positions=10
        )
    assert "cannot exceed max_positions" in str(exc.value)


def test_helper_methods():
    """Test configuration helper methods."""
    from config.config import Settings
    
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        total_capital=1000000.0,
        max_position_size_pct=20.0,
        max_sector_exposure_pct=30.0,
        max_drawdown_pct=2.0
    )
    
    # Test capital calculations
    assert config.get_max_position_value() == 200000.0
    assert config.get_max_sector_value() == 300000.0
    assert config.get_drawdown_threshold() == 20000.0
    
    # Test spread conversions
    assert config.spread_bps_to_percent(20) == 0.002  # 20 bps = 0.2%
    assert config.spread_percent_to_bps(0.002) == 20


def test_is_trading_hours():
    """Test trading hours check."""
    from config.config import Settings
    
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        market_open_hour=9,
        market_open_minute=15,
        market_close_hour=15,
        market_close_minute=30
    )
    
    ist = pytz.timezone('Asia/Kolkata')
    
    # During market hours
    market_time = datetime.now(ist).replace(hour=10, minute=30, second=0, microsecond=0)
    assert config.is_trading_hours(market_time) is True
    
    # Before market open
    before_time = datetime.now(ist).replace(hour=8, minute=0, second=0, microsecond=0)
    assert config.is_trading_hours(before_time) is False
    
    # After market close
    after_time = datetime.now(ist).replace(hour=16, minute=0, second=0, microsecond=0)
    assert config.is_trading_hours(after_time) is False
    
    # At exact close time (should be False due to exclusive upper bound)
    close_time = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)
    assert config.is_trading_hours(close_time) is False


def test_can_place_new_trades():
    """Test new trade placement window."""
    from config.config import Settings
    
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        market_open_hour=9,
        market_open_minute=15,
        market_close_hour=15,
        market_close_minute=30,
        no_trade_start_minutes=15,
        no_trade_end_minutes=15
    )
    
    ist = pytz.timezone('Asia/Kolkata')
    
    # Within trading window (10:30 - well past opening buffer)
    trade_time = datetime.now(ist).replace(hour=10, minute=30, second=0, microsecond=0)
    assert config.can_place_new_trades(trade_time) is True
    
    # During opening buffer (9:20 - within 15 min of open)
    opening_buffer = datetime.now(ist).replace(hour=9, minute=20, second=0, microsecond=0)
    assert config.can_place_new_trades(opening_buffer) is False
    
    # During closing buffer (15:20 - within 15 min of close)
    closing_buffer = datetime.now(ist).replace(hour=15, minute=20, second=0, microsecond=0)
    assert config.can_place_new_trades(closing_buffer) is False


def test_required_fields():
    """Test that required fields are enforced."""
    from config.config import Settings
    
    # Missing broker_api_key
    with pytest.raises(ValidationError) as exc:
        Settings(broker_access_token="test")
    assert "broker_api_key" in str(exc.value)
    
    # Missing broker_access_token
    with pytest.raises(ValidationError) as exc:
        Settings(broker_api_key="test")
    assert "broker_access_token" in str(exc.value)


def test_directory_creation():
    """Test that necessary directories are created."""
    from config.config import Settings
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "data" / "test.db"
        log_path = Path(tmpdir) / "logs" / "test.log"
        
        config = Settings(
            broker_api_key="test",
            broker_access_token="test",
            state_db_path=db_path,
            log_file_path=log_path
        )
        
        assert db_path.parent.exists()
        assert log_path.parent.exists()


def test_log_level_validation():
    """Test log level validation."""
    from config.config import Settings
    
    # Valid log levels
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        config = Settings(
            broker_api_key="test",
            broker_access_token="test",
            log_level=level
        )
        assert config.log_level == level
    
    # Case insensitive
    config = Settings(
        broker_api_key="test",
        broker_access_token="test",
        log_level='info'
    )
    assert config.log_level == 'INFO'
    
    # Invalid log level
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            log_level='INVALID'
        )
    assert "must be one of" in str(exc.value)


def test_environment_variable_loading():
    """Test loading from environment variables."""
    import os
    from config.config import load_settings, reset_settings
    
    # Set environment variables
    os.environ['TRADING_BROKER_API_KEY'] = 'test_key_from_env'
    os.environ['TRADING_BROKER_ACCESS_TOKEN'] = 'test_token_from_env'
    os.environ['TRADING_MAX_POSITIONS'] = '8'
    os.environ['TRADING_DRY_RUN_MODE'] = 'true'
    
    try:
        reset_settings()
        config = load_settings()
        
        assert config.broker_api_key == 'test_key_from_env'
        assert config.broker_access_token == 'test_token_from_env'
        assert config.max_positions == 8
        assert config.dry_run_mode is True
        
    finally:
        # Cleanup
        del os.environ['TRADING_BROKER_API_KEY']
        del os.environ['TRADING_BROKER_ACCESS_TOKEN']
        del os.environ['TRADING_MAX_POSITIONS']
        del os.environ['TRADING_DRY_RUN_MODE']
        reset_settings()


def test_no_extra_fields():
    """Test that extra fields are rejected."""
    from config.config import Settings
    
    with pytest.raises(ValidationError) as exc:
        Settings(
            broker_api_key="test",
            broker_access_token="test",
            unknown_field="should_fail"
        )
    assert "extra" in str(exc.value).lower()


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
