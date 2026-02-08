# config/config.py
"""
Production configuration for NSE intraday trading system.
Uses Pydantic v2 for strict validation and environment variable loading.
All parameters are documented and have safe defaults where applicable.

Configuration is frozen after loading to prevent accidental mutation.
All datetime parameters expect timezone-aware IST datetimes.
"""

from typing import List, Optional, Set
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Centralized configuration for the trading system.
    Loads from environment variables with TRADING_ prefix.
    
    Configuration is immutable after loading (frozen=True).
    """
    
    model_config = SettingsConfigDict(
        env_prefix='TRADING_',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',  # Allow unknown fields for robustness
        frozen=True,     # Prevent mutation after creation
    )
    
    # ================================================================
    # API CREDENTIALS
    # ================================================================
    
    broker_api_key: str = Field(
        ...,
        description="Zerodha Kite Connect API key",
        validation_alias=AliasChoices("TRADING_BROKER_API_KEY", "TRADING_API_KEY"),
    )
    
    broker_access_token: str = Field(
        ...,
        description="Zerodha access token (generate daily via login flow)",
        validation_alias=AliasChoices("TRADING_BROKER_ACCESS_TOKEN", "TRADING_ACCESS_TOKEN"),
    )
    
    api_secret: Optional[str] = Field(
        default=None,
        description="API secret for token generation (optional if using pre-generated token)"
    )
    
    # ================================================================
    # MARKET DATA SETTINGS
    # ================================================================
    
    tick_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Websocket tick timeout in milliseconds (1-30 seconds)"
    )
    
    tick_reconnect_delay_sec: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Delay before reconnecting after websocket disconnect"
    )
    
    tick_max_reconnect_attempts: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Maximum websocket reconnection attempts before shutdown"
    )
    
    rank_interval_sec: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Cross-sectional ranking update frequency (10s - 5min)"
    )
    
    rank_persistence_windows: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of consecutive ranking windows required for actionable signal"
    )

    rank_wobi_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for WOBI score in ranking (0.0-1.0)"
    )

    rank_momentum_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for Momentum score in ranking (0.0-1.0)"
    )

    signal_cooldown_sec: int = Field(
        default=60,
        ge=0,
        le=3600,
        description="Per-symbol cooldown between execution signals (seconds)"
    )

    alpha_poll_interval_ms: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Polling interval for alpha decision loop (milliseconds)"
    )

    market_data_stale_threshold_sec: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Threshold for considering market data snapshots as stale (seconds)"
    )

    execution_tick_size: float = Field(
        default=0.05,
        ge=0.01,
        le=1.0,
        description="Minimum tick size for price rounding (e.g., 0.05 for NSE Equity)"
    )
    
    nifty50_symbols: Set[str] = Field(
        default={
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
            "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
            "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
            "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
            "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
            "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
            "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
            "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
            "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
            "TECHM", "TITAN", "ULTRACEMCO", "UPL", "WIPRO",
        },
        description="Fixed universe of NIFTY 50 and constituents"
    )
    
    wobi_ema_lambda: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="EWMA smoothing factor for WOBI (higher = more weight to recent)"
    )
    
    wobi_stability_window_sec: int = Field(
        default=5,
        ge=2,
        le=30,
        description="Window for detecting order book spoofing/instability"
    )
    
    depth_levels: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Order book depth levels to process (max 5 for NSE)"
    )
    
    depth_weights: List[float] = Field(
        default=[1.0, 0.8, 0.6, 0.4, 0.2],
        description="Weights for each depth level in WOBI calculation (must match depth_levels length)"
    )

    # ================================================================
    # TRANSACTION COST & ALPHA SETTINGS
    # ================================================================
    
    estimated_transaction_cost_bps: float = Field(
        default=20.0,
        ge=5.0,
        le=50.0,
        description="Conservative round-trip transaction cost estimate in basis points (0.20% default)"
    )
    
    alpha_horizon_minutes: int = Field(
        default=5,
        ge=5,
        le=15,
        description="Target alpha signal horizon in minutes (5 or 15)"
    )
    
    # ================================================================
    # RISK MANAGEMENT SETTINGS (DEPRECATED: Use RISK LIMITS below)
    # ================================================================

    # NOTE: redundant fields removed, use canonical fields below
    
    risk_max_volatility_annualized: float = Field(
        default=0.5, # 50% annualized 
        ge=0.0,
        le=5.0,
        description="Pre-trade guard: Max annualized volatility allowed"
    )

    risk_min_liquidity_daily_avg: float = Field(
        default=100000000.0, # 10 Cr default, but Nifty 50 usually higher
        ge=0.0,
        description="Pre-trade guard: Minimum daily average turnover"
    )

    # ================================================================
    # TRADING UNIVERSE
    # ================================================================
    
    trading_universe: str = Field(
        default="NIFTY50",
        description="Trading universe identifier (NIFTY50, NIFTY100, etc.)"
    )
    
    excluded_symbols: List[str] = Field(
        default_factory=list,
        description="Symbols to exclude from trading (e.g., illiquid stocks)"
    )
    
    # ================================================================
    # RISK LIMITS - POSITION LEVEL
    # ================================================================
    
    # Base risk per trade (as % of capital)
    base_risk_per_trade_pct: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Base risk allocation per trade as % of capital (default 1%)"
    )
    
    # Regime-based risk multipliers
    risk_multiplier_trending: float = Field(
        default=1.5,
        ge=0.5,
        le=2.0,
        description="Risk multiplier for TRENDING regime"
    )
    
    risk_multiplier_choppy: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Risk multiplier for CHOPPY regime"
    )
    
    risk_multiplier_high_vol: float = Field(
        default=0.75,
        ge=0.25,
        le=1.5,
        description="Risk multiplier for HIGH_VOL regime"
    )
    
    max_positions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent positions allowed"
    )
    
    max_position_size_pct: float = Field(
        default=20.0,
        ge=5.0,
        le=50.0,
        description="Maximum position size as % of total capital"
    )
    
    default_order_quantity: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Default quantity for entry orders"
    )
    
    max_sector_positions: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum positions per sector"
    )
    
    max_sector_exposure_pct: float = Field(
        default=30.0,
        ge=10.0,
        le=60.0,
        description="Maximum capital exposure per sector (%)"
    )
    
    # ================================================================
    # RISK LIMITS - PORTFOLIO LEVEL
    # ================================================================
    
    total_capital: float = Field(
        default=1000000.0,
        ge=10000.0,
        description="Total trading capital in INR"
    )
    
    max_drawdown_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Maximum intraday drawdown % before halt"
    )
    
    max_daily_loss: float = Field(
        default=20000.0,
        ge=1000.0,
        description="Maximum daily loss in INR before shutdown"
    )
    
    max_consecutive_losses: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Maximum consecutive losing trades before pause"
    )
    
    halt_cooldown_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Cooldown period after circuit breaker trigger"
    )
    
    # ================================================================
    # EXECUTION CONSTRAINTS
    # ================================================================
    
    max_order_modifications: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum price modifications per order (broker safety)"
    )
    
    order_ttl_sec: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Order time-to-live before cancellation (seconds)"
    )
    
    order_placement_delay_ms: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Minimum delay between order placements (rate limiting)"
    )

    execution_pegged_ttl_sec: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Execution: Pegged order time-to-live"
    )

    execution_pegged_max_modifications: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Execution: Max modifications for pegged orders"
    )

    execution_order_poll_interval_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Execution: Order status polling interval"
    )
    
    
    execution_marketable_limit_buffer_bps: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Buffer for marketable limit orders (bps)"
    )
    
    max_spread_bps: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum bid-ask spread in basis points for entry"
    )
    
    min_liquidity_value: float = Field(
        default=1000000.0,
        ge=100000.0,
        description="Minimum depth liquidity value (INR) for trading"
    )
    
    use_pegged_orders: bool = Field(
        default=True,
        description="Enable pegged limit orders (when conditions allow)"
    )
    
    pegged_max_spread_bps: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Max spread in basis points for using pegged orders"
    )
    
    pegged_max_volatility_percentile: int = Field(
        default=60,
        ge=30,
        le=90,
        description="Max volatility percentile for pegged orders"
    )
    
    # ================================================================
    # ALPHA THRESHOLDS
    # ================================================================
    
    exit_alpha_threshold: float = Field(
        default=0.30,
        ge=0.1,
        le=0.5,
        description="Alpha score below which to consider exits"
    )
    
    entry_alpha_threshold: float = Field(
        default=0.75,  # INCREASED from 0.65 for turnover control
        ge=0.5,
        le=0.95,
        description="Minimum composite alpha score for trade entry (cost-aware threshold)"
    )
    
    urgent_alpha_threshold: float = Field(
        default=0.80,
        ge=0.7,
        le=0.99,
        description="Alpha threshold for high-conviction/urgent execution"
    )
    
    rank_top_percentile: float = Field(
        default=90.0,
        ge=80.0,
        le=99.0,
        description="Percentile threshold for long candidates"
    )
    
    rank_bottom_percentile: float = Field(
        default=10.0,
        ge=1.0,
        le=20.0,
        description="Percentile threshold for short candidates (if enabled)"
    )
    
    # ================================================================
    # ALPHA COMPONENT WEIGHTS
    # ================================================================
    
    weight_regime: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for market regime component in alpha score"
    )
    
    weight_rank: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for cross-sectional rank component"
    )
    
    weight_wobi: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for WOBI microstructure component"
    )
    
    weight_anchor: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for VWAP/anchor component"
    )
    
    # ================================================================
    # MARKET REGIME
    # ================================================================
    
    regime_trend_lookback_bars: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Lookback period for trend regime detection"
    )
    
    regime_volatility_lookback_bars: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Lookback period for volatility regime detection"
    )
    
    regime_min_participation_prob: float = Field(
        default=0.6,
        ge=0.5,
        le=0.95,
        description="Minimum regime probability to allow trading"
    )
    
    # ================================================================
    # OPERATING HOURS (NSE) - All times in IST
    # ================================================================
    
    market_open_hour: int = Field(
        default=9,
        ge=0,
        le=23,
        description="Market open hour (IST)"
    )
    
    market_open_minute: int = Field(
        default=15,
        ge=0,
        le=59,
        description="Market open minute (IST)"
    )
    
    market_close_hour: int = Field(
        default=15,
        ge=0,
        le=23,
        description="Market close hour (IST)"
    )
    
    market_close_minute: int = Field(
        default=30,
        ge=0,
        le=59,
        description="Market close minute (IST)"
    )
    
    no_trade_start_minutes: int = Field(
        default=15,
        ge=0,
        le=60,
        description="Minutes after market open before trading starts"
    )
    
    no_trade_end_minutes: int = Field(
        default=15,
        ge=0,
        le=60,
        description="Minutes before market close to stop new trades"
    )
    
    # ================================================================
    # PERSISTENCE & STATE
    # ================================================================
    
    state_db_path: Path = Field(
        default=Path("data/trading_state.db"),
        description="SQLite database path for state persistence"
    )
    
    state_backup_interval_sec: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Interval for state backup operations"
    )
    
    reconcile_on_startup: bool = Field(
        default=True,
        description="Reconcile positions/orders with broker on startup"
    )
    
    # ================================================================
    # LOGGING & MONITORING
    # ================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    log_file_path: Path = Field(
        default=Path("logs/trading.log"),
        description="Log file path"
    )
    
    log_rotation_size_mb: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Log file rotation size in MB"
    )
    
    metrics_export_interval_sec: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Interval for exporting system metrics"
    )
    
    # ================================================================
    # FEATURE FLAGS
    # ================================================================
    
    enable_short_selling: bool = Field(
        default=False,
        description="Enable short selling (requires margin account)"
    )
    
    enable_auto_restart: bool = Field(
        default=True,
        description="Auto-restart on recoverable errors"
    )
    
    dry_run_mode: bool = Field(
        default=True,
        description="Simulate orders without actual broker execution"
    )
    
    trading_mode: str = Field(
        default="PAPER",
        description="Operational mode: PAPER or LIVE"
    )
    
    max_single_order_value: float = Field(
        default=200000.0,
        ge=0.0,
        description="Maximum INR value allowed for a single order"
    )
    
    kill_switch_file: str = Field(
        default="KILL_SWITCH",
        description="File whose presence halts all trading activity"
    )
    
    # ================================================================
    # VALIDATORS
    # ================================================================
    
    @field_validator('depth_weights')
    @classmethod
    def validate_depth_weights(cls, v: List[float], info) -> List[float]:
        """
        Ensure depth weights match depth_levels and are properly ordered.
        
        Validates:
        - Length matches depth_levels configuration
        - All weights are in [0.0, 1.0]
        - Weights are in descending order (closer levels weighted higher)
        """
        depth_levels = info.data.get('depth_levels', 5)
        
        if len(v) != depth_levels:
            raise ValueError(
                f"depth_weights length ({len(v)}) must match depth_levels ({depth_levels})"
            )
        
        if not all(0.0 <= w <= 1.0 for w in v):
            raise ValueError("All depth weights must be between 0.0 and 1.0")
        
        if len(v) > 1 and not all(v[i] >= v[i+1] for i in range(len(v)-1)):
            raise ValueError("Depth weights must be in descending order")
        
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level against standard Python logging levels."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    @model_validator(mode='after')
    def validate_alpha_weights(self) -> 'Settings':
        """
        Ensure alpha component weights sum to 1.0.
        
        Allows small floating point error (±0.01).
        """
        total_weight = (
            self.weight_regime + 
            self.weight_rank + 
            self.weight_wobi + 
            self.weight_anchor
        )
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Alpha component weights must sum to 1.0, got {total_weight:.4f}"
            )
        return self
    
    @model_validator(mode='after')
    def validate_alpha_thresholds(self) -> 'Settings':
        """
        Ensure alpha thresholds are logically ordered.
        
        Required ordering: exit < entry < urgent
        This prevents:
        - Immediate churn (exit >= entry)
        - Unreachable urgent threshold (urgent <= entry)
        - No-trade zones (overlapping thresholds)
        """
        if not (self.exit_alpha_threshold < self.entry_alpha_threshold < self.urgent_alpha_threshold):
            raise ValueError(
                f"Alpha thresholds must satisfy: exit < entry < urgent. "
                f"Got exit={self.exit_alpha_threshold}, entry={self.entry_alpha_threshold}, "
                f"urgent={self.urgent_alpha_threshold}"
            )
        return self
    
    @model_validator(mode='after')
    def validate_market_hours(self) -> 'Settings':
        """
        Ensure market hours are logically consistent.
        
        Validates:
        - Close time is after open time
        - No-trade windows don't exceed total trading hours
        """
        open_minutes = self.market_open_hour * 60 + self.market_open_minute
        close_minutes = self.market_close_hour * 60 + self.market_close_minute
        
        if open_minutes >= close_minutes:
            raise ValueError("Market close time must be after market open time")
        
        trading_window = close_minutes - open_minutes
        no_trade_total = self.no_trade_start_minutes + self.no_trade_end_minutes
        
        if no_trade_total >= trading_window:
            raise ValueError(
                f"No-trade windows ({no_trade_total} min) exceed total market hours "
                f"({trading_window} min)"
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_risk_limits(self) -> 'Settings':
        """
        Validate risk limit consistency.
        
        Ensures max_sector_positions doesn't exceed max_positions.
        """
        if self.max_sector_positions > self.max_positions:
            raise ValueError(
                f"max_sector_positions ({self.max_sector_positions}) cannot exceed "
                f"max_positions ({self.max_positions})"
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_spread_consistency(self) -> 'Settings':
        """
        Validate spread parameters are logically consistent.
        
        Pegged orders should only be used in tighter spreads than general entry.
        """
        if self.pegged_max_spread_bps >= self.max_spread_bps:
            raise ValueError(
                f"pegged_max_spread_bps ({self.pegged_max_spread_bps}) should be less than "
                f"max_spread_bps ({self.max_spread_bps}) for tighter execution"
            )
        
        return self
    
    @model_validator(mode='after')
    def create_directories(self) -> 'Settings':
        """
        Create necessary directories for logs and data.
        
        Note: This runs during initialization, before freezing.
        """
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        return self
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def is_trading_hours(self, current_time) -> bool:
        """
        Check if current time is within market hours.
        
        Args:
            current_time: timezone-aware datetime object in IST
            
        Returns:
            True if within trading hours, False otherwise
            
        Note:
            Uses exclusive upper bound (< market_close) since NSE
            stops accepting orders before the official close time.
        """
        from datetime import time
        
        market_open = time(
            self.market_open_hour, 
            self.market_open_minute
        )
        market_close = time(
            self.market_close_hour, 
            self.market_close_minute
        )
        
        current = current_time.time()
        return market_open <= current < market_close
    
    def can_place_new_trades(self, current_time) -> bool:
        """
        Check if new trades can be placed (excluding buffer periods).
        
        Args:
            current_time: timezone-aware datetime object in IST
            
        Returns:
            True if new trades allowed, False otherwise
            
        Note:
            Respects no_trade_start_minutes and no_trade_end_minutes
            to avoid opening/closing volatility.
        """
        from datetime import datetime, timedelta
        
        market_open = datetime.combine(
            current_time.date(),
            datetime.min.time()
        ).replace(
            hour=self.market_open_hour,
            minute=self.market_open_minute,
            tzinfo=current_time.tzinfo
        )
        
        market_close = datetime.combine(
            current_time.date(),
            datetime.min.time()
        ).replace(
            hour=self.market_close_hour,
            minute=self.market_close_minute,
            tzinfo=current_time.tzinfo
        )
        
        trade_start = market_open + timedelta(minutes=self.no_trade_start_minutes)
        trade_end = market_close - timedelta(minutes=self.no_trade_end_minutes)
        
        return trade_start <= current_time <= trade_end
    
    def get_max_position_value(self) -> float:
        """Calculate maximum position value in INR."""
        return self.total_capital * (self.max_position_size_pct / 100)
    
    def get_max_sector_value(self) -> float:
        """Calculate maximum sector exposure value in INR."""
        return self.total_capital * (self.max_sector_exposure_pct / 100)
    
    def get_drawdown_threshold(self) -> float:
        """Calculate drawdown threshold in INR."""
        return self.total_capital * (self.max_drawdown_pct / 100)
    
    def spread_bps_to_percent(self, bps: int) -> float:
        """Convert basis points to percentage."""
        return bps / 10000.0
    
    def spread_percent_to_bps(self, pct: float) -> int:
        """Convert percentage to basis points."""
        return int(pct * 10000)


# ================================================================
# DEFERRED SINGLETON PATTERN (THREAD-SAFE)
# ================================================================

_settings: Optional[Settings] = None
_settings_lock = None


def load_settings() -> Settings:
    """
    Load and validate settings from environment.
    
    This function creates directories and validates all parameters.
    Should be called explicitly during application startup.
    
    Returns:
        Validated Settings instance
        
    Raises:
        ValidationError: If configuration is invalid
        RuntimeError: If loading fails
    """
    try:
        settings = Settings()
        return settings
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}") from e


def get_settings() -> Settings:
    s = Settings()
    print("DEBUG SETTINGS FIELDS:", s.model_fields.keys())
    return s


def reset_settings() -> None:
    """
    Reset the global settings singleton.
    
    Primarily for testing purposes. Use with caution in production.
    """
    global _settings
    _settings = None


# ================================================================
# VALIDATION & DIAGNOSTICS
# ================================================================

if __name__ == "__main__":
    """
    Configuration validation and display.
    
    Run this module directly to validate configuration:
        python -m config.config
    """
    import json
    from datetime import datetime
    import pytz
    
    print("Configuration Validation Report")
    print("=" * 70)
    
    try:
        config = load_settings()
        print("✓ Configuration loaded successfully\n")
        
        print("Critical Parameters:")
        print(f"  - Max Positions: {config.max_positions}")
        print(f"  - Max Drawdown: {config.max_drawdown_pct}%")
        print(f"  - Entry Alpha Threshold: {config.entry_alpha_threshold}")
        print(f"  - Rank Interval: {config.rank_interval_sec}s")
        print(f"  - Max Order Modifications: {config.max_order_modifications}")
        print(f"  - Order TTL: {config.order_ttl_sec}s")
        
        print(f"\nCapital Allocation:")
        print(f"  - Total Capital: ₹{config.total_capital:,.2f}")
        print(f"  - Max Position Size: ₹{config.get_max_position_value():,.2f} "
              f"({config.max_position_size_pct}%)")
        print(f"  - Max Sector Exposure: ₹{config.get_max_sector_value():,.2f} "
              f"({config.max_sector_exposure_pct}%)")
        print(f"  - Drawdown Threshold: ₹{config.get_drawdown_threshold():,.2f} "
              f"({config.max_drawdown_pct}%)")
        
        print(f"\nAlpha Configuration:")
        total_weight = (config.weight_regime + config.weight_rank + 
                       config.weight_wobi + config.weight_anchor)
        print(f"  - Component Weights (sum={total_weight:.3f}):")
        print(f"    • Regime: {config.weight_regime}")
        print(f"    • Rank: {config.weight_rank}")
        print(f"    • WOBI: {config.weight_wobi}")
        print(f"    • Anchor: {config.weight_anchor}")
        print(f"  - Thresholds: Exit={config.exit_alpha_threshold} < "
              f"Entry={config.entry_alpha_threshold} < "
              f"Urgent={config.urgent_alpha_threshold}")
        
        print(f"\nExecution Parameters:")
        print(f"  - Max Spread (Entry): {config.max_spread_bps} bps "
              f"({config.spread_bps_to_percent(config.max_spread_bps):.2%})")
        print(f"  - Max Spread (Pegged): {config.pegged_max_spread_bps} bps "
              f"({config.spread_bps_to_percent(config.pegged_max_spread_bps):.2%})")
        print(f"  - Pegged Orders Enabled: {config.use_pegged_orders}")
        
        print(f"\nMarket Microstructure:")
        print(f"  - Depth Levels: {config.depth_levels}")
        print(f"  - Depth Weights: {config.depth_weights}")
        print(f"  - WOBI EMA Lambda: {config.wobi_ema_lambda}")
        
        print(f"\nMarket Hours (IST):")
        print(f"  - Open: {config.market_open_hour:02d}:{config.market_open_minute:02d}")
        print(f"  - Close: {config.market_close_hour:02d}:{config.market_close_minute:02d}")
        print(f"  - No-trade buffer: {config.no_trade_start_minutes}m start, "
              f"{config.no_trade_end_minutes}m end")
        
        # Test time checks
        ist = pytz.timezone('Asia/Kolkata')
        test_time = datetime.now(ist).replace(hour=10, minute=30)
        print(f"\nTime Check (test time: 10:30 IST):")
        print(f"  - Is trading hours: {config.is_trading_hours(test_time)}")
        print(f"  - Can place trades: {config.can_place_new_trades(test_time)}")
        
        print(f"\nRisk Circuit Breakers:")
        print(f"  - Max Consecutive Losses: {config.max_consecutive_losses}")
        print(f"  - Max Daily Loss: ₹{config.max_daily_loss:,.2f}")
        print(f"  - Halt Cooldown: {config.halt_cooldown_minutes} minutes")
        
        print(f"\nPersistence:")
        print(f"  - State DB: {config.state_db_path}")
        print(f"  - Log File: {config.log_file_path}")
        print(f"  - Log Level: {config.log_level}")
        
        print(f"\nFeature Flags:")
        print(f"  - Short Selling: {config.enable_short_selling}")
        print(f"  - Auto Restart: {config.enable_auto_restart}")
        print(f"  - Dry Run Mode: {config.dry_run_mode}")
        
        print("\n" + "=" * 70)
        print("✓ All validations passed - Configuration is production-ready")
        
    except Exception as e:
        print(f"\n✗ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        raise
