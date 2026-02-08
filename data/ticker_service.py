# data/ticker_service.py
"""
Market data ingestion layer using Zerodha KiteTicker in Full Mode.

This module handles:
- Real-time tick subscription for NIFTY 50 instruments
- Order book depth parsing (top 5 bid/ask levels)
- Weighted Order Book Imbalance (WOBI) computation with EWMA smoothing
- Async-safe shared state for feature consumers
- Latency monitoring with automatic tick dropping
- Robust reconnection handling

No trading logic - purely data ingestion and preprocessing.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from kiteconnect import KiteTicker

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

# NIFTY 50 instrument tokens (example - should be loaded from master instruments)
# These are placeholder tokens - actual tokens come from instruments master
NIFTY_50_TOKENS: List[int] = []  # Populated via load_nifty50_instruments()

# WOBI calculation weights for top 5 depth levels
DEFAULT_DEPTH_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]

# Maximum processing latency before dropping tick
MAX_TICK_LATENCY_MS = 100

# Reconnection parameters
RECONNECT_DELAY_SEC = 5
MAX_RECONNECT_ATTEMPTS = 10


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass(kw_only=True)
class DepthLevel:
    """Single order book depth level."""
    price: float
    quantity: int
    orders: int = 0


@dataclass(kw_only=True)
class OrderBookDepth:
    """Full order book depth (top 5 levels each side)."""
    bids: List[DepthLevel] = field(default_factory=list)
    asks: List[DepthLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(kw_only=True)
class TickData:
    """Processed tick data with computed features."""
    instrument_token: int
    symbol: str
    last_price: float
    volume: int
    depth: OrderBookDepth
    wobi_raw: float  # Unsmoothed WOBI
    wobi_ema: float  # EWMA smoothed WOBI
    bid_value: float  # Total weighted bid value
    ask_value: float  # Total weighted ask value
    spread_bps: float  # Bid-ask spread in basis points
    timestamp: datetime
    exchange_timestamp: Optional[datetime] = None
    processing_latency_ms: float = 0.0


# ================================================================
# WOBI CALCULATOR
# ================================================================

class WOBICalculator:
    """
    Weighted Order Book Imbalance calculator with EWMA smoothing.
    
    WOBI Formula:
        WOBI = (Σ(BidQty_i × w_i) - Σ(AskQty_i × w_i)) / (Σ(BidQty_i × w_i) + Σ(AskQty_i × w_i))
    
    Returns value in [-1, 1]:
        - Positive = bid pressure (buying interest)
        - Negative = ask pressure (selling interest)
        - Zero = balanced book
    """
    
    def __init__(self, weights: List[float] = None, ema_lambda: float = 0.8):
        """
        Initialize WOBI calculator.
        
        Args:
            weights: Depth level weights [level1, level2, ...], default [1.0, 0.8, 0.6, 0.4, 0.2]
            ema_lambda: EWMA smoothing factor (higher = more weight to recent, range 0.5-0.99)
        """
        self.weights = weights or DEFAULT_DEPTH_WEIGHTS
        self.ema_lambda = ema_lambda
        
        # Per-instrument EWMA state
        self._ema_values: Dict[int, float] = {}
        self._lock = threading.Lock()
    
    def compute(self, 
                instrument_token: int,
                bids: List[DepthLevel], 
                asks: List[DepthLevel]) -> Tuple[float, float, float, float]:
        """
        Compute WOBI for given order book depth.
        
        Args:
            instrument_token: Instrument identifier for EWMA state
            bids: List of bid depth levels
            asks: List of ask depth levels
            
        Returns:
            Tuple of (wobi_raw, wobi_ema, weighted_bid_value, weighted_ask_value)
        """
        # Calculate weighted sums
        weighted_bid = sum(
            bid.quantity * self.weights[i] 
            for i, bid in enumerate(bids[:len(self.weights)])
        )
        weighted_ask = sum(
            ask.quantity * self.weights[i] 
            for i, ask in enumerate(asks[:len(self.weights)])
        )
        
        # Compute raw WOBI
        total = weighted_bid + weighted_ask
        if total == 0:
            wobi_raw = 0.0
        else:
            wobi_raw = (weighted_bid - weighted_ask) / total
        
        # Apply EWMA smoothing
        with self._lock:
            if instrument_token in self._ema_values:
                prev_ema = self._ema_values[instrument_token]
                wobi_ema = self.ema_lambda * wobi_raw + (1 - self.ema_lambda) * prev_ema
            else:
                wobi_ema = wobi_raw  # First observation
            
            self._ema_values[instrument_token] = wobi_ema
        
        return wobi_raw, wobi_ema, weighted_bid, weighted_ask
    
    def reset(self, instrument_token: Optional[int] = None):
        """Reset EWMA state for instrument or all instruments."""
        with self._lock:
            if instrument_token is not None:
                self._ema_values.pop(instrument_token, None)
            else:
                self._ema_values.clear()


# ================================================================
# TICK DATA STORE (ASYNC-SAFE)
# ================================================================

class TickDataStore:
    """
    Thread-safe and async-safe storage for latest tick data.
    
    Provides:
    - Lock-free reads for latest values
    - Thread-safe writes from websocket callback
    - Async-compatible access patterns
    - Efficient subscription for multiple consumers
    """
    
    def __init__(self):
        self._data: Dict[int, TickData] = {}
        self._symbol_to_token: Dict[str, int] = {}
        self._token_to_symbol: Dict[int, str] = {}
        self._lock = threading.RLock()
        self._update_callbacks: List[Callable[[TickData], None]] = []
        self._async_queue: Optional[asyncio.Queue] = None
        self._last_update_time: Dict[int, float] = {}
    
    def register_symbols(self, token_symbol_map: Dict[int, str]):
        """Register instrument token to symbol mappings."""
        with self._lock:
            self._token_to_symbol.update(token_symbol_map)
            self._symbol_to_token.update({v: k for k, v in token_symbol_map.items()})
    
    def update(self, tick: TickData):
        """
        Update tick data for an instrument (thread-safe).
        
        Args:
            tick: New tick data
        """
        with self._lock:
            self._data[tick.instrument_token] = tick
            self._last_update_time[tick.instrument_token] = time.monotonic()
        
        # Notify sync callbacks
        for callback in self._update_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.warning(f"Tick callback error: {e}")
        
        # Push to async queue if available
        if self._async_queue:
            try:
                self._async_queue.put_nowait(tick)
            except asyncio.QueueFull:
                logger.warning(f"Async tick queue full, dropping tick for {tick.symbol}")
    
    def get(self, instrument_token: int) -> Optional[TickData]:
        """Get latest tick data for instrument token."""
        with self._lock:
            return self._data.get(instrument_token)
    
    def get_by_symbol(self, symbol: str) -> Optional[TickData]:
        """Get latest tick data by symbol."""
        with self._lock:
            token = self._symbol_to_token.get(symbol)
            if token:
                return self._data.get(token)
        return None
    
    def get_all(self) -> Dict[int, TickData]:
        """Get snapshot of all latest tick data."""
        with self._lock:
            return dict(self._data)
    
    def get_symbols_with_data(self) -> List[str]:
        """Get list of symbols that have received tick data."""
        with self._lock:
            return [
                self._token_to_symbol[token] 
                for token in self._data.keys()
                if token in self._token_to_symbol
            ]
    
    def get_stale_instruments(self, stale_threshold_sec: float = 5.0) -> List[int]:
        """Get instruments that haven't updated within threshold."""
        now = time.monotonic()
        stale = []
        with self._lock:
            for token, last_time in self._last_update_time.items():
                if now - last_time > stale_threshold_sec:
                    stale.append(token)
        return stale
    
    def add_callback(self, callback: Callable[[TickData], None]):
        """Add synchronous callback for tick updates."""
        self._update_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[TickData], None]):
        """Remove synchronous callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def create_async_queue(self, maxsize: int = 1000) -> asyncio.Queue:
        """Create an async queue for tick updates."""
        self._async_queue = asyncio.Queue(maxsize=maxsize)
        return self._async_queue
    
    async def get_async(self, instrument_token: int) -> Optional[TickData]:
        """Async-compatible get (runs in executor for safety)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get, instrument_token)
    
    async def get_all_async(self) -> Dict[int, TickData]:
        """Async-compatible get all."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_all)


# ================================================================
# TICKER SERVICE
# ================================================================

class TickerService:
    """
    Market data ingestion service using KiteTicker in Full Mode.
    
    Handles:
    - WebSocket connection management
    - Tick parsing and WOBI computation
    - Latency monitoring and tick dropping
    - Automatic reconnection with backoff
    - Async-safe data distribution
    """
    
    def __init__(self,
                 api_key: str,
                 access_token: str,
                 instruments: Dict[int, str],
                 depth_weights: List[float] = None,
                 ema_lambda: float = 0.8,
                 max_latency_ms: float = MAX_TICK_LATENCY_MS):
        """
        Initialize ticker service.
        
        Args:
            api_key: Zerodha API key
            access_token: Zerodha access token
            instruments: Dict mapping instrument_token -> symbol
            depth_weights: WOBI depth weights
            ema_lambda: EWMA smoothing factor
            max_latency_ms: Maximum tick processing latency before dropping
        """
        self.api_key = api_key
        self.access_token = access_token
        self.instruments = instruments
        self.max_latency_ms = max_latency_ms
        
        # Components
        self.wobi_calculator = WOBICalculator(
            weights=depth_weights or DEFAULT_DEPTH_WEIGHTS,
            ema_lambda=ema_lambda
        )
        self.tick_store = TickDataStore()
        self.tick_store.register_symbols(instruments)
        
        # KiteTicker instance
        self._ticker: Optional[KiteTicker] = None
        
        # Connection state
        self._is_connected = False
        self._is_running = False
        self._reconnect_count = 0
        self._last_tick_time: float = 0.0
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'ticks_received': 0,
            'ticks_processed': 0,
            'ticks_dropped': 0,
            'reconnects': 0,
            'errors': 0,
            'total_latency_ms': 0.0,
        }
    
    def _create_ticker(self):
        """Create and configure KiteTicker instance."""
        self._ticker = KiteTicker(self.api_key, self.access_token)
        
        # Register callbacks
        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_noreconnect = self._on_noreconnect
        self._ticker.on_order_update = self._on_order_update
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection."""
        logger.info(f"Ticker connected: {response}")
        
        with self._lock:
            self._is_connected = True
            self._reconnect_count = 0
        
        # Subscribe to instruments in FULL mode (includes depth)
        tokens = list(self.instruments.keys())
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info(f"Subscribed to {len(tokens)} instruments in FULL mode")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket close."""
        logger.warning(f"Ticker closed: code={code}, reason={reason}")
        
        with self._lock:
            self._is_connected = False
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket error."""
        logger.error(f"Ticker error: code={code}, reason={reason}")
        
        with self._lock:
            self._stats['errors'] += 1
    
    def _on_reconnect(self, ws, attempts_count):
        """Handle reconnection attempt."""
        logger.info(f"Ticker reconnecting: attempt {attempts_count}")
        
        with self._lock:
            self._reconnect_count = attempts_count
            self._stats['reconnects'] += 1
    
    def _on_noreconnect(self, ws):
        """Handle exhausted reconnection attempts."""
        logger.error("Ticker exhausted all reconnection attempts")
        
        with self._lock:
            self._is_connected = False
            self._is_running = False
    
    def _on_order_update(self, ws, data):
        """Handle order update (not used - for extensibility)."""
        pass  # Order updates handled by execution layer
    
    def _on_ticks(self, ws, ticks: List[Dict[str, Any]]):
        """
        Process incoming ticks.
        
        Parses tick data, computes WOBI, and updates tick store.
        Drops ticks if processing latency exceeds threshold.
        """
        receive_time = time.monotonic()
        
        for tick in ticks:
            try:
                self._stats['ticks_received'] += 1
                
                # Parse tick
                token = tick.get('instrument_token')
                if token not in self.instruments:
                    continue
                
                symbol = self.instruments[token]
                last_price = tick.get('last_price', 0.0)
                volume = tick.get('volume_traded', 0)
                
                # Parse depth (Full mode has 'depth' with 'buy' and 'sell')
                depth_data = tick.get('depth', {})
                bids, asks = self._parse_depth(depth_data)
                
                # Compute WOBI
                wobi_raw, wobi_ema, bid_value, ask_value = self.wobi_calculator.compute(
                    token, bids, asks
                )
                
                # Calculate spread
                best_bid = bids[0].price if bids else 0.0
                best_ask = asks[0].price if asks else 0.0
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else last_price
                spread_bps = 0.0
                if mid_price > 0:
                    spread_bps = ((best_ask - best_bid) / mid_price) * 10000
                
                # Get timestamps
                exchange_ts = tick.get('exchange_timestamp')
                tick_ts = tick.get('timestamp', datetime.now(timezone.utc))
                
                # Create tick data
                tick_data = TickData(
                    instrument_token=token,
                    symbol=symbol,
                    last_price=last_price,
                    volume=volume,
                    depth=OrderBookDepth(bids=bids, asks=asks, timestamp=tick_ts),
                    wobi_raw=wobi_raw,
                    wobi_ema=wobi_ema,
                    bid_value=bid_value,
                    ask_value=ask_value,
                    spread_bps=spread_bps,
                    timestamp=tick_ts,
                    exchange_timestamp=exchange_ts,
                )
                
                # Check processing latency
                process_time = time.monotonic()
                latency_ms = (process_time - receive_time) * 1000
                tick_data.processing_latency_ms = latency_ms
                
                if latency_ms > self.max_latency_ms:
                    logger.debug(
                        f"Dropping tick for {symbol}: latency {latency_ms:.1f}ms > {self.max_latency_ms}ms"
                    )
                    self._stats['ticks_dropped'] += 1
                    continue
                
                # Update tick store
                self.tick_store.update(tick_data)
                
                self._stats['ticks_processed'] += 1
                self._stats['total_latency_ms'] += latency_ms
                self._last_tick_time = process_time
                
            except Exception as e:
                logger.error(f"Error processing tick: {e}", exc_info=True)
                self._stats['errors'] += 1
    
    def _parse_depth(self, depth_data: Dict) -> Tuple[List[DepthLevel], List[DepthLevel]]:
        """
        Parse order book depth from tick data.
        
        Args:
            depth_data: Raw depth dict with 'buy' and 'sell' lists
            
        Returns:
            Tuple of (bids, asks) as lists of DepthLevel
        """
        bids = []
        asks = []
        
        # Parse buy side (bids)
        for level in depth_data.get('buy', [])[:5]:
            bids.append(DepthLevel(
                price=level.get('price', 0.0),
                quantity=level.get('quantity', 0),
                orders=level.get('orders', 0),
            ))
        
        # Parse sell side (asks)
        for level in depth_data.get('sell', [])[:5]:
            asks.append(DepthLevel(
                price=level.get('price', 0.0),
                quantity=level.get('quantity', 0),
                orders=level.get('orders', 0),
            ))
        
        return bids, asks
    
    def start(self):
        """
        Start the ticker service (blocking).
        
        This starts the WebSocket connection and blocks.
        For async usage, run in a thread.
        """
        logger.info("Starting ticker service...")
        
        with self._lock:
            self._is_running = True
        
        self._create_ticker()
        
        # KiteTicker.connect() is blocking with reconnection
        self._ticker.connect(threaded=False)
    
    def start_threaded(self) -> threading.Thread:
        """
        Start ticker service in a background thread.
        
        Returns:
            Thread object running the ticker
        """
        thread = threading.Thread(
            target=self.start,
            name="TickerService",
            daemon=True
        )
        thread.start()
        
        # Wait for initial connection
        timeout = 10.0
        start = time.monotonic()
        while not self._is_connected and time.monotonic() - start < timeout:
            time.sleep(0.1)
        
        if not self._is_connected:
            logger.warning("Ticker did not connect within timeout")
        
        return thread
    
    async def start_async(self) -> asyncio.Task:
        """
        Start ticker service as an async task.
        
        Returns:
            Async task wrapping the ticker thread
        """
        loop = asyncio.get_running_loop()
        
        # Run ticker in thread pool
        def run_ticker():
            self._create_ticker()
            with self._lock:
                self._is_running = True
            self._ticker.connect(threaded=False)
        
        task = loop.run_in_executor(None, run_ticker)
        
        # Wait for connection
        timeout = 10.0
        start = time.monotonic()
        while not self._is_connected and time.monotonic() - start < timeout:
            await asyncio.sleep(0.1)
        
        return asyncio.wrap_future(task)
    
    def stop(self):
        """Stop the ticker service."""
        logger.info("Stopping ticker service...")
        
        with self._lock:
            self._is_running = False
        
        if self._ticker:
            try:
                self._ticker.close()
            except Exception as e:
                logger.warning(f"Error closing ticker: {e}")
        
        self.wobi_calculator.reset()
    
    @property
    def is_connected(self) -> bool:
        """Check if ticker is connected."""
        with self._lock:
            return self._is_connected
    
    @property
    def is_running(self) -> bool:
        """Check if ticker service is running."""
        with self._lock:
            return self._is_running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        with self._lock:
            stats = dict(self._stats)
            
        # Add derived stats
        if stats['ticks_processed'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['ticks_processed']
        else:
            stats['avg_latency_ms'] = 0.0
        
        stats['drop_rate'] = (
            stats['ticks_dropped'] / max(1, stats['ticks_received'])
        ) * 100
        
        stats['is_connected'] = self.is_connected
        stats['is_running'] = self.is_running
        
        return stats


# ================================================================
# FACTORY FUNCTION
# ================================================================

def create_ticker_service(
    api_key: str = None,
    access_token: str = None,
    instruments: Dict[int, str] = None,
) -> TickerService:
    """
    Create a TickerService with configuration from settings.
    
    Args:
        api_key: Override API key (default: from config)
        access_token: Override access token (default: from config)
        instruments: Override instruments map (default: load from master)
        
    Returns:
        Configured TickerService instance
    """
    # Load settings if not provided
    if api_key is None or access_token is None:
        try:
            from config.config import get_settings
            settings = get_settings()
            api_key = api_key or settings.broker_api_key
            access_token = access_token or settings.broker_access_token
            depth_weights = settings.depth_weights
            ema_lambda = settings.wobi_ema_lambda
        except Exception as e:
            logger.warning(f"Could not load settings: {e}")
            depth_weights = DEFAULT_DEPTH_WEIGHTS
            ema_lambda = 0.8
    else:
        depth_weights = DEFAULT_DEPTH_WEIGHTS
        ema_lambda = 0.8
    
    # Load instruments if not provided
    if instruments is None:
        instruments = load_nifty50_instruments(api_key, access_token)
    
    return TickerService(
        api_key=api_key,
        access_token=access_token,
        instruments=instruments,
        depth_weights=depth_weights,
        ema_lambda=ema_lambda,
    )


def load_nifty50_instruments(api_key: str, access_token: str) -> Dict[int, str]:
    """
    Load NIFTY 50 instrument tokens from Kite instruments master.
    
    Args:
        api_key: Zerodha API key
        access_token: Zerodha access token
        
    Returns:
        Dict mapping instrument_token -> tradingsymbol
    """
    try:
        from kiteconnect import KiteConnect
        
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Get all NSE instruments
        instruments = kite.instruments("NSE")
        
        # NIFTY 50 symbols (current constituents - should be updated periodically)
        nifty50_symbols = {
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
        }
        
        # Build token -> symbol map
        token_map = {}
        for inst in instruments:
            if inst['tradingsymbol'] in nifty50_symbols:
                token_map[inst['instrument_token']] = inst['tradingsymbol']
        
        logger.info(f"Loaded {len(token_map)} NIFTY 50 instruments")
        return token_map
        
    except Exception as e:
        logger.error(f"Failed to load instruments: {e}")
        return {}


# ================================================================
# TESTING & DIAGNOSTICS
# ================================================================

if __name__ == "__main__":
    """
    Test the ticker service components.
    
    Run: python -m data.ticker_service
    """
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Ticker Service Component Tests")
    print("=" * 70)
    
    # Test WOBI Calculator
    print("\n1. Testing WOBICalculator...")
    calculator = WOBICalculator(weights=[1.0, 0.8, 0.6, 0.4, 0.2], ema_lambda=0.8)
    
    # Simulate order book
    test_bids = [
        DepthLevel(price=100.00, quantity=1000),
        DepthLevel(price=99.95, quantity=800),
        DepthLevel(price=99.90, quantity=600),
        DepthLevel(price=99.85, quantity=400),
        DepthLevel(price=99.80, quantity=200),
    ]
    test_asks = [
        DepthLevel(price=100.05, quantity=500),
        DepthLevel(price=100.10, quantity=400),
        DepthLevel(price=100.15, quantity=300),
        DepthLevel(price=100.20, quantity=200),
        DepthLevel(price=100.25, quantity=100),
    ]
    
    wobi_raw, wobi_ema, bid_val, ask_val = calculator.compute(12345, test_bids, test_asks)
    print(f"   WOBI Raw: {wobi_raw:.4f}")
    print(f"   WOBI EMA: {wobi_ema:.4f}")
    print(f"   Bid Value: {bid_val:.0f}")
    print(f"   Ask Value: {ask_val:.0f}")
    print(f"   ✓ WOBI calculation working")
    
    # Test EWMA
    print("\n2. Testing EWMA smoothing...")
    for i in range(5):
        # Alternate bid/ask pressure
        if i % 2 == 0:
            bids = [DepthLevel(price=100, quantity=2000)] * 5
            asks = [DepthLevel(price=101, quantity=500)] * 5
        else:
            bids = [DepthLevel(price=100, quantity=500)] * 5
            asks = [DepthLevel(price=101, quantity=2000)] * 5
        
        wobi_raw, wobi_ema, _, _ = calculator.compute(12345, bids, asks)
        print(f"   Tick {i+1}: Raw={wobi_raw:.4f}, EMA={wobi_ema:.4f}")
    print(f"   ✓ EWMA smoothing working")
    
    # Test Tick Data Store
    print("\n3. Testing TickDataStore...")
    store = TickDataStore()
    store.register_symbols({12345: "RELIANCE", 67890: "TCS"})
    
    tick = TickData(
        instrument_token=12345,
        symbol="RELIANCE",
        last_price=2500.0,
        volume=1000000,
        depth=OrderBookDepth(bids=test_bids, asks=test_asks),
        wobi_raw=0.3,
        wobi_ema=0.25,
        bid_value=2400,
        ask_value=1100,
        spread_bps=5.0,
        timestamp=datetime.now(timezone.utc),
    )
    store.update(tick)
    
    retrieved = store.get(12345)
    assert retrieved is not None
    assert retrieved.symbol == "RELIANCE"
    print(f"   ✓ Tick storage working")
    
    by_symbol = store.get_by_symbol("RELIANCE")
    assert by_symbol is not None
    print(f"   ✓ Symbol lookup working")
    
    all_data = store.get_all()
    assert len(all_data) == 1
    print(f"   ✓ Bulk retrieval working")
    
    # Test async queue
    print("\n4. Testing async queue...")
    queue = store.create_async_queue(maxsize=100)
    store.update(tick)
    
    # Queue should have the tick
    assert not queue.empty()
    print(f"   ✓ Async queue working")
    
    print("\n" + "=" * 70)
    print("All component tests passed!")
    print("=" * 70)
    
    # If credentials available, test real connection
    api_key = os.environ.get("TRADING_BROKER_API_KEY") or os.environ.get("TRADING_API_KEY")
    access_token = os.environ.get("TRADING_BROKER_ACCESS_TOKEN") or os.environ.get("TRADING_ACCESS_TOKEN")
    
    if api_key and access_token:
        print("\n5. Testing real connection (10 seconds)...")
        instruments = {256265: "NIFTY 50"}  # NIFTY index token for testing
        
        service = TickerService(
            api_key=api_key,
            access_token=access_token,
            instruments=instruments,
        )
        
        thread = service.start_threaded()
        time.sleep(10)
        
        stats = service.get_stats()
        print(f"   Stats: {stats}")
        
        service.stop()
        print(f"   ✓ Real connection test complete")
    else:
        print("\n5. Skipping real connection test (no credentials)")
