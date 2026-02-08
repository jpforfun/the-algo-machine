"""
Crash-safe state persistence manager for live trading engine.

Uses SQLite with WAL mode for atomic transactions and concurrent reads.
All blocking operations are offloaded to thread pool executors.
Implements reconciliation logic to handle broker vs. local state mismatches.

Key features:
- ACID-compliant transactions
- WAL mode for better concurrency
- Connection pooling for thread safety
- Automatic state compression and cleanup
- Comprehensive error recovery
- Async-safe via dedicated thread pool executor
"""

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

IST = timezone(timedelta(hours=5, minutes=30))  # IST timezone

# ================================================================
# DATA MODELS
# ================================================================

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_FILL = "PARTIAL_FILL"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFYING = "MODIFYING"
    CANCEL_PENDING = "CANCEL_PENDING"
    FALLBACK_TRIGGERED = "FALLBACK_TRIGGERED"

class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"

class RiskFlag(str, Enum):
    """Risk flag enumeration."""
    TRADING_HALTED = "TRADING_HALTED"
    DRAWDOWN_BREACHED = "DRAWDOWN_BREACHED"
    MAX_DAILY_LOSS_BREACHED = "MAX_DAILY_LOSS_BREACHED"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    SYSTEM_ERROR = "SYSTEM_ERROR"

@dataclass(kw_only=True)
class Order:
    """Order data model."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    price: float
    trigger_price: Optional[float]
    status: OrderStatus
    order_timestamp: datetime
    last_updated: datetime
    broker_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None

@dataclass(kw_only=True)
class Position:
    """Position data model."""
    position_id: str
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    last_updated: datetime
    status: PositionStatus
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    tags: Optional[Dict[str, Any]] = None

@dataclass(kw_only=True)
class PnLRecord:
    """PnL record data model."""
    record_id: str
    date: str  # YYYY-MM-DD
    symbol: str
    realized_pnl: float
    unrealized_pnl: float
    timestamp: datetime
    position_ids: List[str]

@dataclass(kw_only=True)
class StateSnapshot:
    """Complete state snapshot."""
    positions: List[Position]
    open_orders: List[Order]
    risk_flags: List[RiskFlag]
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    timestamp: datetime
    
    @property
    def daily_total_pnl(self) -> float:
        """Total daily PnL (realized + unrealized)."""
        return self.daily_realized_pnl + self.daily_unrealized_pnl

# ================================================================
# DATABASE SCHEMA
# ================================================================

SCHEMA_VERSION = 2

INIT_SQL = f"""
-- Enable WAL mode for better concurrency and crash safety
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 3000;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Risk flags table
CREATE TABLE IF NOT EXISTS risk_flags (
    flag_id TEXT PRIMARY KEY,
    flag_type TEXT NOT NULL CHECK (flag_type IN (
        'TRADING_HALTED',
        'DRAWDOWN_BREACHED', 
        'MAX_DAILY_LOSS_BREACHED',
        'MAX_CONSECUTIVE_LOSSES',
        'SYSTEM_ERROR'
    )),
    is_active BOOLEAN NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    metadata TEXT
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    current_price REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED', 'PARTIALLY_CLOSED')),
    exit_price REAL,
    exit_time TIMESTAMP,
    realized_pnl REAL DEFAULT 0.0,
    unrealized_pnl REAL DEFAULT 0.0,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    broker_order_id TEXT UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    filled_quantity INTEGER NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0 AND filled_quantity <= quantity),
    price REAL NOT NULL CHECK (price > 0),
    trigger_price REAL,
    status TEXT NOT NULL CHECK (status IN (
        'PENDING', 'OPEN', 'PARTIAL_FILL', 'COMPLETE', 
        'CANCELLED', 'REJECTED', 'MODIFYING', 
        'CANCEL_PENDING', 'FALLBACK_TRIGGERED'
    )),
    order_timestamp TIMESTAMP NOT NULL,
    last_updated TIMESTAMP NOT NULL,
    parent_order_id TEXT REFERENCES orders(order_id) ON DELETE SET NULL,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- PnL tracking table (daily aggregated)
CREATE TABLE IF NOT EXISTS daily_pnl (
    record_id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    timestamp TIMESTAMP NOT NULL,
    position_ids TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE(date, symbol)
);

-- Order-Position mapping table
CREATE TABLE IF NOT EXISTS order_position_map (
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    position_id TEXT NOT NULL REFERENCES positions(position_id) ON DELETE CASCADE,
    allocation_quantity INTEGER NOT NULL CHECK (allocation_quantity > 0),
    allocation_price REAL NOT NULL CHECK (allocation_price > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (order_id, position_id)
);

-- Trade history table (closed positions)
CREATE TABLE IF NOT EXISTS trade_history (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    realized_pnl REAL NOT NULL,
    holding_period_seconds INTEGER,
    tags TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- State audit log
CREATE TABLE IF NOT EXISTS state_audit_log (
    audit_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_state TEXT,
    new_state TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    metadata TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_status_timestamp ON orders(status, order_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);
CREATE INDEX IF NOT EXISTS idx_risk_flags_active ON risk_flags(is_active);
CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol_date ON trade_history(symbol, exit_time);
CREATE INDEX IF NOT EXISTS idx_state_audit_timestamp ON state_audit_log(timestamp);
"""

# ================================================================
# CONNECTION POOL
# ================================================================

class ConnectionPool:
    """
    Thread-safe SQLite connection pool.
    
    Each thread gets its own connection to avoid threading issues.
    Connections are stored in thread-local storage and tracked globally
    so they can all be closed during shutdown.
    """
    
    def __init__(self, db_path: Path, timeout: int = 10):
        self.db_path = db_path
        self.timeout = timeout
        self._local = threading.local()
        self._lock = threading.RLock()
        self._all_connections: Dict[int, sqlite3.Connection] = {}  # thread_id -> connection
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        thread_id = threading.get_ident()
        if not hasattr(self._local, 'connection'):
            with self._lock:
                # Double-checked locking
                if not hasattr(self._local, 'connection'):
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=self.timeout,
                        check_same_thread=False  # We ensure thread safety via pool
                    )
                    # Configure connection
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = NORMAL")
                    conn.execute("PRAGMA busy_timeout = 3000")
                    self._local.connection = conn
                    self._all_connections[thread_id] = conn  # Track for close_all
                    logger.debug(f"Created new connection for thread {thread_id}")
        
        return self._local.connection
    
    def close_all(self):
        """Close ALL connections across all threads."""
        with self._lock:
            # Close all tracked connections from all threads
            for thread_id, conn in list(self._all_connections.items()):
                try:
                    conn.close()
                    logger.debug(f"Closed connection for thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Error closing connection for thread {thread_id}: {e}")
            self._all_connections.clear()
            
            # Also clean up current thread's local reference
            if hasattr(self._local, 'connection'):
                delattr(self._local, 'connection')


# ================================================================
# STATE MANAGER
# ================================================================

class StateManager:
    """
    Crash-safe state persistence manager.
    
    Thread-safe and async-safe. All blocking database operations
    are executed in a dedicated thread pool executor to avoid blocking async loops.
    """
    
    def __init__(self, db_path: Path, max_history_days: int = 30):
        """
        Initialize state manager.
        
        Args:
            db_path: Path to SQLite database file
            max_history_days: Maximum days to keep trade history
        """
        self.db_path = db_path
        self.max_history_days = max_history_days
        
        # Dedicated thread pool executor (FIXED CRITICAL #1)
        self._executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="StateManagerThread"
        )
        
        self.pool = ConnectionPool(db_path)
        self._initialized = False
        self._lock = threading.RLock()
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Don't run cleanup on init - only after market close (FIXED MEDIUM #4)
        logger.info(f"State manager initialized at {self.db_path}")
    
    # ================================================================
    # DATABASE INITIALIZATION
    # ================================================================
    
    def _initialize_database(self):
        """Initialize database schema and tables."""
        try:
            with self._transaction() as conn:
                # Execute schema creation
                conn.executescript(INIT_SQL)
                
                # Record schema version
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
                
                # Check schema version (FIXED MEDIUM #6)
                version_result = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                
                if version_result:
                    current_version = version_result[0]
                    if current_version > SCHEMA_VERSION:
                        raise RuntimeError(
                            f"Database schema version ({current_version}) is newer than "
                            f"application version ({SCHEMA_VERSION}). "
                            f"Please upgrade the application."
                        )
                    elif current_version < SCHEMA_VERSION:
                        self._run_migrations(current_version, SCHEMA_VERSION, conn)
                
                self._initialized = True
                logger.info(f"Database initialized, version {SCHEMA_VERSION}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _run_migrations(self, from_version: int, to_version: int, conn: sqlite3.Connection):
        """Run database migrations.
        
        Note: SQLite does not support ALTER TABLE ADD CHECK for adding constraints
        to existing tables. CHECK constraints can only be defined at table creation.
        The filled_quantity <= quantity constraint is enforced via application-level
        validation in save_order() and update_order_status().
        """
        logger.info(f"Running migrations from version {from_version} to {to_version}")
        
        # Migration 1 -> 2
        if from_version < 2 <= to_version:
            # Add missing indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_parent_id 
                ON orders(parent_order_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_entry_time 
                ON positions(entry_time)
            """)
            
            # NOTE: SQLite does not support ALTER TABLE ADD CHECK for existing tables.
            # The filled_quantity <= quantity constraint is enforced at application level
            # in save_order() and update_order_status() methods.
            logger.info("Migration V1->V2: Added indexes. CHECK constraint enforced at app level.")
        
        # Update schema version
        conn.execute(
            "UPDATE schema_version SET version = ? WHERE version = ?",
            (to_version, from_version)
        )
    
    # ================================================================
    # TRANSACTION MANAGEMENT
    # ================================================================
    
    @contextmanager
    def _transaction(self, isolation_level: str = None, write: bool = False):
        """
        Context manager for database transactions.
        
        Args:
            isolation_level: Optional explicit isolation level
            write: If True, use IMMEDIATE transaction for writes (FIXED MEDIUM #1)
        
        Ensures atomic commits or rollbacks on exceptions.
        """
        conn = self.pool.get_connection()
        
        # Set appropriate isolation level
        if isolation_level:
            conn.execute(f"BEGIN {isolation_level}")
        elif write:
            conn.execute("BEGIN IMMEDIATE")  # Better for concurrent writes
        else:
            conn.execute("BEGIN")  # Default for reads
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            # Reset connection state
            pass
    
    # ================================================================
    # ASYNC EXECUTOR (FIXED CRITICAL #1)
    # ================================================================
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking function in dedicated thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )
    
    # ================================================================
    # POSITION MANAGEMENT
    # ================================================================
    
    def save_position(self, position: Position) -> str:
        """
        Save or update a position.
        
        Args:
            position: Position object to save
            
        Returns:
            Position ID
            
        Raises:
            ValueError: If position data is invalid
        """
        # Validate input
        if position.quantity <= 0:
            raise ValueError(f"Invalid quantity: {position.quantity}")
        
        if position.entry_price <= 0:
            raise ValueError(f"Invalid entry price: {position.entry_price}")
        
        with self._transaction(write=True) as conn:
            # Convert tags to JSON
            tags_json = json.dumps(position.tags) if position.tags else None
            
            # Check if position exists
            existing = conn.execute(
                "SELECT position_id FROM positions WHERE position_id = ?",
                (position.position_id,)
            ).fetchone()
            
            if existing:
                # Update existing position
                conn.execute("""
                    UPDATE positions 
                    SET symbol = ?, side = ?, quantity = ?, entry_price = ?,
                        current_price = ?, status = ?, exit_price = ?,
                        exit_time = ?, realized_pnl = ?, unrealized_pnl = ?,
                        tags = ?, last_updated = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE position_id = ?
                """, (
                    position.symbol,
                    position.side.value,
                    position.quantity,
                    position.entry_price,
                    position.current_price,
                    position.status.value,
                    position.exit_price,
                    position.exit_time.isoformat() if position.exit_time else None,
                    position.realized_pnl,
                    position.unrealized_pnl,
                    tags_json,
                    position.last_updated.isoformat(),
                    position.position_id
                ))
                
                # Log update (skip for mark-to-market updates) (FIXED MEDIUM #3)
                if position.status != PositionStatus.OPEN or abs(position.unrealized_pnl) > 10:
                    self._log_audit("UPDATE", "position", position.position_id, None, position._asdict(), conn)
                logger.debug(f"Updated position {position.position_id}")
            else:
                # Insert new position
                conn.execute("""
                    INSERT INTO positions (
                        position_id, symbol, side, quantity, entry_price,
                        current_price, entry_time, last_updated, status,
                        exit_price, exit_time, realized_pnl, unrealized_pnl, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    position.position_id,
                    position.symbol,
                    position.side.value,
                    position.quantity,
                    position.entry_price,
                    position.current_price,
                    position.entry_time.isoformat(),
                    position.last_updated.isoformat(),
                    position.status.value,
                    position.exit_price,
                    position.exit_time.isoformat() if position.exit_time else None,
                    position.realized_pnl,
                    position.unrealized_pnl,
                    tags_json
                ))
                
                # Log creation
                self._log_audit("CREATE", "position", position.position_id, None, position._asdict(), conn)
                logger.info(f"Created new position {position.position_id}")
            
            # Update daily PnL (using trade_history for realized) (FIXED CRITICAL #3)
            self._update_daily_pnl(position.symbol, conn)
            
            return position.position_id
    
    async def save_position_async(self, position: Position) -> str:
        """Async wrapper for save_position."""
        return await self._run_in_executor(self.save_position, position)
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Retrieve a position by ID."""
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM positions WHERE position_id = ?",
                (position_id,)
            ).fetchone()
            
            if row:
                return self._row_to_position(row)
            return None
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions, optionally filtered by symbol."""
        with self._transaction() as conn:
            if symbol:
                rows = conn.execute("""
                    SELECT * FROM positions 
                    WHERE status = 'OPEN' AND symbol = ?
                    ORDER BY entry_time
                """, (symbol,))
            else:
                rows = conn.execute("""
                    SELECT * FROM positions 
                    WHERE status = 'OPEN'
                    ORDER BY entry_time
                """)
            
            return [self._row_to_position(row) for row in rows]
    
    async def get_open_positions_async(self, symbol: Optional[str] = None) -> List[Position]:
        """Async wrapper for get_open_positions."""
        return await self._run_in_executor(self.get_open_positions, symbol)
    
    def close_position(
        self, 
        position_id: str, 
        exit_price: float,
        exit_time: datetime,
        realized_pnl: float,
        partial_quantity: Optional[int] = None
    ) -> bool:
        """
        Close a position (fully or partially) and move it to trade history.
        
        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_time: Exit timestamp (must be timezone-aware in UTC)
            realized_pnl: Realized PnL for this trade closure
            partial_quantity: If provided, partial closure quantity
            
        Returns:
            True if successful, False if position not found
        """
        with self._transaction(write=True) as conn:
            # Get position details
            position_row = conn.execute(
                "SELECT * FROM positions WHERE position_id = ?",
                (position_id,)
            ).fetchone()
            
            if not position_row:
                logger.warning(f"Position {position_id} not found for closing")
                return False
            
            position = self._row_to_position(position_row)
            
            # Determine closure type
            if partial_quantity and partial_quantity < position.quantity:
                # Partial closure
                new_quantity = position.quantity - partial_quantity
                new_status = PositionStatus.PARTIALLY_CLOSED
                
                # Update position quantity and realized PnL
                conn.execute("""
                    UPDATE positions 
                    SET quantity = ?,
                        realized_pnl = realized_pnl + ?,
                        status = ?,
                        last_updated = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE position_id = ?
                """, (
                    new_quantity,
                    realized_pnl,
                    new_status.value,
                    exit_time.isoformat(),
                    position_id
                ))
                
                # Create trade history record for partial closure
                trade_id = f"TRADE_{uuid.uuid4().hex[:8]}"
                holding_period = int((exit_time - position.entry_time).total_seconds())
                
                conn.execute("""
                    INSERT INTO trade_history (
                        trade_id, position_id, symbol, side, quantity,
                        entry_price, exit_price, entry_time, exit_time,
                        realized_pnl, holding_period_seconds, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    position_id,
                    position.symbol,
                    position.side.value,
                    partial_quantity,
                    position.entry_price,
                    exit_price,
                    position.entry_time.isoformat(),
                    exit_time.isoformat(),
                    realized_pnl,
                    holding_period,
                    json.dumps(position.tags) if position.tags else None
                ))
                
                logger.info(f"Partially closed position {position_id}: "
                           f"{partial_quantity}/{position.quantity} shares, PnL: {realized_pnl:.2f}")
                
            else:
                # Full closure
                # Create trade history record
                trade_id = f"TRADE_{uuid.uuid4().hex[:8]}"
                holding_period = int((exit_time - position.entry_time).total_seconds())
                
                conn.execute("""
                    INSERT INTO trade_history (
                        trade_id, position_id, symbol, side, quantity,
                        entry_price, exit_price, entry_time, exit_time,
                        realized_pnl, holding_period_seconds, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    position_id,
                    position.symbol,
                    position.side.value,
                    position.quantity,
                    position.entry_price,
                    exit_price,
                    position.entry_time.isoformat(),
                    exit_time.isoformat(),
                    realized_pnl,
                    holding_period,
                    json.dumps(position.tags) if position.tags else None
                ))
                
                # Update position status to closed
                conn.execute("""
                    UPDATE positions 
                    SET status = 'CLOSED',
                        exit_price = ?,
                        exit_time = ?,
                        realized_pnl = ?,
                        current_price = ?,
                        last_updated = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE position_id = ?
                """, (
                    exit_price,
                    exit_time.isoformat(),
                    realized_pnl,
                    exit_price,
                    exit_time.isoformat(),
                    position_id
                ))
                
                logger.info(f"Closed position {position_id} with PnL: {realized_pnl:.2f}")
            
            # Update daily PnL
            self._update_daily_pnl(position.symbol, conn)
            
            # Log closure
            self._log_audit("CLOSE", "position", position_id, position._asdict(), None, conn)
            
            return True
    
    # ================================================================
    # ORDER MANAGEMENT
    # ================================================================
    
    def save_order(self, order: Order) -> str:
        """
        Save or update an order.
        
        Args:
            order: Order object to save
            
        Returns:
            Order ID
            
        Raises:
            ValueError: If order data is invalid
        """
        # Validate input
        if order.quantity <= 0:
            raise ValueError(f"Invalid quantity: {order.quantity}")
        
        if order.price <= 0:
            raise ValueError(f"Invalid price: {order.price}")
        
        if order.filled_quantity > order.quantity:
            raise ValueError(f"Filled quantity ({order.filled_quantity}) exceeds order quantity ({order.quantity})")
        
        with self._transaction(write=True) as conn:
            # Convert tags to JSON
            tags_json = json.dumps(order.tags) if order.tags else None
            
            # Check if order exists
            existing = conn.execute(
                "SELECT order_id FROM orders WHERE order_id = ?",
                (order.order_id,)
            ).fetchone()
            
            if existing:
                # Update existing order
                conn.execute("""
                    UPDATE orders 
                    SET broker_order_id = ?, symbol = ?, side = ?, quantity = ?,
                        filled_quantity = ?, price = ?, trigger_price = ?,
                        status = ?, last_updated = ?, parent_order_id = ?,
                        tags = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                """, (
                    order.broker_order_id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.filled_quantity,
                    order.price,
                    order.trigger_price,
                    order.status.value,
                    order.last_updated.isoformat(),
                    order.parent_order_id,
                    tags_json,
                    order.order_id
                ))
                
                # Log update (skip for status-only updates) (FIXED MEDIUM #3)
                if order.status not in [OrderStatus.OPEN, OrderStatus.PARTIAL_FILL]:
                    self._log_audit("UPDATE", "order", order.order_id, None, order._asdict(), conn)
                logger.debug(f"Updated order {order.order_id}")
            else:
                # Insert new order
                conn.execute("""
                    INSERT INTO orders (
                        order_id, broker_order_id, symbol, side, quantity,
                        filled_quantity, price, trigger_price, status,
                        order_timestamp, last_updated, parent_order_id, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.order_id,
                    order.broker_order_id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.filled_quantity,
                    order.price,
                    order.trigger_price,
                    order.status.value,
                    order.order_timestamp.isoformat(),
                    order.last_updated.isoformat(),
                    order.parent_order_id,
                    tags_json
                ))
                
                # Log creation
                self._log_audit("CREATE", "order", order.order_id, None, order._asdict(), conn)
                logger.info(f"Created new order {order.order_id}")
            
            return order.order_id
    
    async def save_order_async(self, order: Order) -> str:
        """Async wrapper for save_order."""
        return await self._run_in_executor(self.save_order, order)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders (PENDING, OPEN, PARTIAL_FILL, MODIFYING)."""
        with self._transaction() as conn:
            if symbol:
                rows = conn.execute("""
                    SELECT * FROM orders 
                    WHERE status IN ('PENDING', 'OPEN', 'PARTIAL_FILL', 'MODIFYING')
                    AND symbol = ?
                    ORDER BY order_timestamp
                """, (symbol,))
            else:
                rows = conn.execute("""
                    SELECT * FROM orders 
                    WHERE status IN ('PENDING', 'OPEN', 'PARTIAL_FILL', 'MODIFYING')
                    ORDER BY order_timestamp
                """)
            
            return [self._row_to_order(row) for row in rows]
    
    async def get_open_orders_async(self, symbol: Optional[str] = None) -> List[Order]:
        """Async wrapper for get_open_orders."""
        return await self._run_in_executor(self.get_open_orders, symbol)
    
    def update_order_status(
        self, 
        order_id: str, 
        status: OrderStatus,
        filled_quantity: Optional[int] = None,
        broker_order_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order ID to update
            status: New status
            filled_quantity: Updated filled quantity (if any)
            broker_order_id: Broker order ID (if available)
            timestamp: Update timestamp (must be timezone-aware in UTC, defaults to now)
            
        Returns:
            True if successful, False if order not found
        """
        update_time = timestamp or datetime.now(timezone.utc)
        
        with self._transaction(write=True) as conn:
            # Get existing order
            existing = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,)
            ).fetchone()
            
            if not existing:
                logger.warning(f"Order {order_id} not found for status update")
                return False
            
            # Validate filled quantity
            if filled_quantity is not None:
                if filled_quantity > existing['quantity']:
                    logger.error(f"Filled quantity {filled_quantity} exceeds order quantity {existing['quantity']}")
                    return False
            
            # Build update query
            updates = ["status = ?", "last_updated = ?", "updated_at = CURRENT_TIMESTAMP"]
            params = [status.value, update_time.isoformat()]
            
            if filled_quantity is not None:
                updates.append("filled_quantity = ?")
                params.append(filled_quantity)
            
            if broker_order_id is not None:
                updates.append("broker_order_id = ?")
                params.append(broker_order_id)
            
            params.append(order_id)  # WHERE clause parameter
            
            # Execute update
            query = f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?"
            conn.execute(query, params)
            
            # Skip audit log for frequent status updates (FIXED MEDIUM #3)
            if status not in [OrderStatus.OPEN, OrderStatus.PARTIAL_FILL]:
                self._log_audit("UPDATE_STATUS", "order", order_id, None, {
                    "status": status.value,
                    "filled_quantity": filled_quantity
                }, conn)
            
            logger.info(f"Updated order {order_id} to status {status.value}")
            return True
    
    # ================================================================
    # RISK FLAG MANAGEMENT (FIXED MEDIUM #5)
    # ================================================================
    
    def set_risk_flag(self, flag_type: RiskFlag, metadata: Optional[Dict] = None) -> str:
        """
        Set a risk flag as active.
        
        Args:
            flag_type: Type of risk flag
            metadata: Additional metadata about the flag
            
        Returns:
            Flag ID
        """
        flag_id = f"FLAG_{uuid.uuid4().hex[:8]}"
        
        with self._transaction(write=True) as conn:
            # Check if flag already active
            existing = conn.execute("""
                SELECT flag_id FROM risk_flags 
                WHERE flag_type = ? AND is_active = 1
            """, (flag_type.value,)).fetchone()
            
            if existing:
                logger.warning(f"Risk flag {flag_type.value} already active")
                return existing[0]
            
            # Insert new flag
            metadata_json = json.dumps(metadata) if metadata else None
            
            conn.execute("""
                INSERT INTO risk_flags (flag_id, flag_type, metadata)
                VALUES (?, ?, ?)
            """, (flag_id, flag_type.value, metadata_json))
            
            # Log creation
            self._log_audit("CREATE", "risk_flag", flag_id, None, {
                "flag_type": flag_type.value,
                "metadata": metadata
            }, conn)
            
            logger.warning(f"Set risk flag: {flag_type.value}")
            return flag_id
    
    async def set_risk_flag_async(self, flag_type: RiskFlag, metadata: Optional[Dict] = None) -> str:
        """Async wrapper for set_risk_flag."""
        return await self._run_in_executor(self.set_risk_flag, flag_type, metadata)
    
    def clear_risk_flag(self, flag_type: RiskFlag) -> bool:
        """
        Clear a risk flag.
        
        Args:
            flag_type: Type of risk flag to clear
            
        Returns:
            True if flag was cleared, False if not active
        """
        with self._transaction(write=True) as conn:
            # Check if flag exists and is active
            existing = conn.execute("""
                SELECT flag_id FROM risk_flags 
                WHERE flag_type = ? AND is_active = 1
            """, (flag_type.value,)).fetchone()
            
            if not existing:
                logger.warning(f"Risk flag {flag_type.value} not active")
                return False
            
            flag_id = existing[0]
            
            # Deactivate flag
            conn.execute("""
                UPDATE risk_flags 
                SET is_active = 0, resolved_at = CURRENT_TIMESTAMP
                WHERE flag_id = ?
            """, (flag_id,))
            
            # Log deactivation
            self._log_audit("CLEAR", "risk_flag", flag_id, {"flag_type": flag_type.value}, None, conn)
            
            logger.info(f"Cleared risk flag: {flag_type.value}")
            return True
    
    def get_active_risk_flags(self) -> List[RiskFlag]:
        """Get all active risk flags."""
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT flag_type FROM risk_flags WHERE is_active = 1"
            ).fetchall()
            
            return [RiskFlag(row[0]) for row in rows]
    
    async def get_active_risk_flags_async(self) -> List[RiskFlag]:
        """Async wrapper for get_active_risk_flags."""
        return await self._run_in_executor(self.get_active_risk_flags)
    
    def is_trading_halted(self) -> bool:
        """
        Check if trading is halted due to risk flags.
        
        Returns:
            True if trading should be halted, False otherwise
        """
        active_flags = self.get_active_risk_flags()
        
        # Define which flags should halt trading
        halt_flags = {
            RiskFlag.TRADING_HALTED,
            RiskFlag.DRAWDOWN_BREACHED,
            RiskFlag.MAX_DAILY_LOSS_BREACHED,
            RiskFlag.MAX_CONSECUTIVE_LOSSES,
            RiskFlag.SYSTEM_ERROR
        }
        
        return any(flag in active_flags for flag in halt_flags)
    
    async def is_trading_halted_async(self) -> bool:
        """Async wrapper for is_trading_halted."""
        return await self._run_in_executor(self.is_trading_halted)
    
    # ================================================================
    # PnL MANAGEMENT (FIXED CRITICAL #3)
    # ================================================================
    
    def _update_daily_pnl(self, symbol: str, conn: sqlite3.Connection):
        """Update daily PnL for a symbol."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Calculate realized PnL from trade history (FIXED CRITICAL #3)
        realized_result = conn.execute("""
            SELECT COALESCE(SUM(realized_pnl), 0) 
            FROM trade_history 
            WHERE symbol = ? 
            AND DATE(exit_time) = ?
        """, (symbol, today)).fetchone()
        
        realized_pnl = realized_result[0] if realized_result else 0.0
        
        # Calculate unrealized PnL from open positions
        unrealized_result = conn.execute("""
            SELECT COALESCE(SUM(unrealized_pnl), 0)
            FROM positions
            WHERE symbol = ?
            AND status IN ('OPEN', 'PARTIALLY_CLOSED')
        """, (symbol,)).fetchone()
        
        unrealized_pnl = unrealized_result[0] if unrealized_result else 0.0
        
        # Get position IDs for open positions
        position_rows = conn.execute("""
            SELECT position_id FROM positions 
            WHERE symbol = ? AND status IN ('OPEN', 'PARTIALLY_CLOSED')
        """, (symbol,)).fetchall()
        
        position_ids = [row[0] for row in position_rows]
        
        # Insert or update daily PnL
        record_id = f"PNL_{uuid.uuid4().hex[:8]}"
        
        conn.execute("""
            INSERT OR REPLACE INTO daily_pnl 
            (record_id, date, symbol, realized_pnl, unrealized_pnl, timestamp, position_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record_id,
            today,
            symbol,
            realized_pnl,
            unrealized_pnl,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(position_ids) if position_ids else None
        ))
    
    def get_daily_pnl(self, date: Optional[str] = None) -> List[PnLRecord]:
        """
        Get daily PnL records.
        
        Args:
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            List of PnL records
        """
        query_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        with self._transaction() as conn:
            rows = conn.execute("""
                SELECT * FROM daily_pnl WHERE date = ?
            """, (query_date,)).fetchall()
            
            records = []
            for row in rows:
                position_ids = json.loads(row['position_ids']) if row['position_ids'] else []
                records.append(PnLRecord(
                    record_id=row['record_id'],
                    date=row['date'],
                    symbol=row['symbol'],
                    realized_pnl=row['realized_pnl'],
                    unrealized_pnl=row['unrealized_pnl'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    position_ids=position_ids
                ))
            
            return records
    
    def _get_total_daily_pnl_internal(self, conn: sqlite3.Connection) -> Tuple[float, float]:
        """
        Internal method to get daily PnL using an existing connection.
        
        Args:
            conn: Existing database connection within a transaction
            
        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Realized PnL from trade history
        realized_result = conn.execute("""
            SELECT COALESCE(SUM(realized_pnl), 0) 
            FROM trade_history 
            WHERE DATE(exit_time) = ?
        """, (today,)).fetchone()
        
        realized_pnl = realized_result[0] if realized_result else 0.0
        
        # Unrealized PnL from open positions
        unrealized_result = conn.execute("""
            SELECT COALESCE(SUM(unrealized_pnl), 0)
            FROM positions
            WHERE status IN ('OPEN', 'PARTIALLY_CLOSED')
        """).fetchone()
        
        unrealized_pnl = unrealized_result[0] if unrealized_result else 0.0
        
        return realized_pnl, unrealized_pnl
    
    def get_total_daily_pnl(self) -> Tuple[float, float]:
        """
        Get total realized and unrealized PnL for today.
        
        Returns:
            Tuple of (realized_pnl, unrealized_pnl)
        """
        with self._transaction() as conn:
            return self._get_total_daily_pnl_internal(conn)
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """
        Get recent trades from history.
        
        Args:
            limit: Number of trades to fetch
            
        Returns:
            List of trade dictionaries
        """
        with self._transaction() as conn:
            rows = conn.execute("""
                SELECT * FROM trade_history 
                ORDER BY exit_time DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [dict(row) for row in rows]
            
    async def get_recent_trades_async(self, limit: int = 10) -> List[Dict]:
        """Async wrapper for get_recent_trades."""
        return await self._run_in_executor(self.get_recent_trades, limit)

    def get_recent_closed_positions(self, limit: int = 10) -> List[Position]:
        """
        Get recently closed positions (for consecutive loss logic).
        
        Args:
            limit: Number of positions to fetch
            
        Returns:
            List of closed positions ordered by exit time
        """
        with self._transaction() as conn:
            rows = conn.execute("""
                SELECT * FROM positions 
                WHERE status = 'CLOSED'
                ORDER BY exit_time DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [self._row_to_position(row) for row in rows]

    async def get_recent_closed_positions_async(self, limit: int = 10) -> List[Position]:
        """Async wrapper for get_recent_closed_positions."""
        return await self._run_in_executor(self.get_recent_closed_positions, limit)
    
    # ================================================================
    # STATE LOADING & RECONCILIATION (FIXED CRITICAL #4)
    # ================================================================
    
    def load_open_state(self) -> StateSnapshot:
        """
        Load complete open state from database.
        
        Returns:
            StateSnapshot containing all open positions, orders, and risk flags
        """
        with self._transaction() as conn:
            # Load open positions
            position_rows = conn.execute("""
                SELECT * FROM positions 
                WHERE status IN ('OPEN', 'PARTIALLY_CLOSED')
            """).fetchall()
            
            positions = [self._row_to_position(row) for row in position_rows]
            
            # Load open orders
            order_rows = conn.execute("""
                SELECT * FROM orders 
                WHERE status IN ('PENDING', 'OPEN', 'PARTIAL_FILL', 'MODIFYING')
            """).fetchall()
            
            orders = [self._row_to_order(row) for row in order_rows]
            
            # Load active risk flags
            flag_rows = conn.execute(
                "SELECT flag_type FROM risk_flags WHERE is_active = 1"
            ).fetchall()
            
            risk_flags = [RiskFlag(row[0]) for row in flag_rows]
            
            # Get daily PnL (using internal method to avoid nested transaction)
            realized_pnl, unrealized_pnl = self._get_total_daily_pnl_internal(conn)
            
            return StateSnapshot(
                positions=positions,
                open_orders=orders,
                risk_flags=risk_flags,
                daily_realized_pnl=realized_pnl,
                daily_unrealized_pnl=unrealized_pnl,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def load_open_state_async(self) -> StateSnapshot:
        """Async wrapper for load_open_state."""
        return await self._run_in_executor(self.load_open_state)
    
    def reconcile_with_broker(self, broker_positions: List[Dict]) -> Dict[str, Any]:
        """
        Reconcile local state with broker positions.
        
        Args:
            broker_positions: List of positions from broker API
            
        Returns:
            Dictionary containing reconciliation results
        """
        # Load local positions (has its own transaction)
        local_positions = self.get_open_positions()
        
        # Convert to dicts with composite key (FIXED CRITICAL #4)
        local_dict = {}
        for pos in local_positions:
            # Create composite key: symbol_side (e.g., "RELIANCE_LONG")
            key = f"{pos.symbol}_{pos.side.value}"
            
            # Handle multiple positions with same symbol and side
            if key in local_dict:
                # Calculate weighted average price BEFORE updating quantity
                old_qty = local_dict[key]['quantity']
                new_qty = old_qty + pos.quantity
                total_value = (local_dict[key]['entry_price'] * old_qty +
                             pos.entry_price * pos.quantity)
                local_dict[key]['entry_price'] = total_value / new_qty
                local_dict[key]['quantity'] = new_qty
                local_dict[key]['position_ids'].append(pos.position_id)
            else:
                local_dict[key] = {
                    'position_ids': [pos.position_id],
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price
                }
        
        # Process broker positions
        broker_dict = {}
        for pos in broker_positions:
            quantity = pos.get('quantity', 0)
            if abs(quantity) > 0:
                symbol = pos.get('symbol', '')
                side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
                key = f"{symbol}_{side.value}"
                
                if key in broker_dict:
                    # Calculate weighted average price BEFORE updating quantity
                    old_qty = broker_dict[key]['quantity']
                    new_qty = old_qty + abs(quantity)
                    total_value = (broker_dict[key]['entry_price'] * old_qty +
                                 pos.get('average_price', 0.0) * abs(quantity))
                    broker_dict[key]['entry_price'] = total_value / new_qty
                    broker_dict[key]['quantity'] = new_qty
                else:
                    broker_dict[key] = {
                        'side': side,
                        'quantity': abs(quantity),
                        'entry_price': pos.get('average_price', 0.0)
                    }
        
        # Find discrepancies
        missing_positions = []  # In broker but not local
        extra_positions = []    # In local but not broker
        quantity_mismatches = []
        price_mismatches = []
        
        # Check all broker positions
        for key, broker_data in broker_dict.items():
            if key not in local_dict:
                missing_positions.append({
                    'symbol': key.split('_')[0],
                    'side': broker_data['side'].value,
                    'broker_quantity': broker_data['quantity'],
                    'broker_price': broker_data['entry_price']
                })
                continue
            
            local_data = local_dict[key]
            
            # Check quantity mismatch (allow 1% tolerance or 1 unit minimum)
            local_qty = local_data['quantity']
            broker_qty = broker_data['quantity']
            qty_tolerance = max(1, local_qty * 0.01)  # 1% or 1 unit
            
            if abs(local_qty - broker_qty) > qty_tolerance:
                quantity_mismatches.append({
                    'symbol': key.split('_')[0],
                    'side': broker_data['side'].value,
                    'position_ids': local_data['position_ids'],
                    'local_quantity': local_qty,
                    'broker_quantity': broker_qty,
                    'difference': broker_qty - local_qty,
                    'difference_pct': abs(local_qty - broker_qty) / local_qty * 100 if local_qty > 0 else 0
                })
            
            # Check price mismatch (allow 0.1% tolerance)
            local_price = local_data['entry_price']
            broker_price = broker_data['entry_price']
            
            if broker_price > 0 and local_price > 0:
                price_diff_pct = abs(local_price - broker_price) / broker_price * 100
                if price_diff_pct > 0.1:  # 0.1% tolerance
                    price_mismatches.append({
                        'symbol': key.split('_')[0],
                        'side': broker_data['side'].value,
                        'position_ids': local_data['position_ids'],
                        'local_price': local_price,
                        'broker_price': broker_price,
                        'difference_pct': price_diff_pct
                    })
        
        # Check for positions in local but not broker
        for key, local_data in local_dict.items():
            if key not in broker_dict:
                extra_positions.append({
                    'symbol': key.split('_')[0],
                    'side': local_data['side'].value,
                    'position_ids': local_data['position_ids'],
                    'local_quantity': local_data['quantity'],
                    'local_price': local_data['entry_price']
                })
        
        # Log reconciliation results
        if missing_positions or extra_positions or quantity_mismatches or price_mismatches:
            logger.warning(f"Reconciliation found issues: "
                         f"missing={len(missing_positions)}, "
                         f"extra={len(extra_positions)}, "
                         f"qty_mismatch={len(quantity_mismatches)}, "
                         f"price_mismatch={len(price_mismatches)}")
        
        return {
            'missing_positions': missing_positions,
            'extra_positions': extra_positions,
            'quantity_mismatches': quantity_mismatches,
            'price_mismatches': price_mismatches,
            'local_count': len(local_positions),
            'broker_count': len(broker_positions),
            'local_aggregated_count': len(local_dict),
            'broker_aggregated_count': len(broker_dict),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def reconcile_with_broker_async(self, broker_positions: List[Dict]) -> Dict[str, Any]:
        """Async wrapper for reconcile_with_broker."""
        return await self._run_in_executor(self.reconcile_with_broker, broker_positions)
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def _row_to_position(self, row) -> Position:
        """Convert SQLite row to Position object."""
        return Position(
            position_id=row['position_id'],
            symbol=row['symbol'],
            side=PositionSide(row['side']),
            quantity=row['quantity'],
            entry_price=row['entry_price'],
            current_price=row['current_price'],
            entry_time=datetime.fromisoformat(row['entry_time']),
            last_updated=datetime.fromisoformat(row['last_updated']),
            status=PositionStatus(row['status']),
            exit_price=row['exit_price'],
            exit_time=datetime.fromisoformat(row['exit_time']) if row['exit_time'] else None,
            realized_pnl=row['realized_pnl'] or 0.0,
            unrealized_pnl=row['unrealized_pnl'] or 0.0,
            tags=json.loads(row['tags']) if row['tags'] else None
        )
    
    def _row_to_order(self, row) -> Order:
        """Convert SQLite row to Order object."""
        return Order(
            order_id=row['order_id'],
            symbol=row['symbol'],
            side=OrderSide(row['side']),
            quantity=row['quantity'],
            filled_quantity=row['filled_quantity'],
            price=row['price'],
            trigger_price=row['trigger_price'],
            status=OrderStatus(row['status']),
            order_timestamp=datetime.fromisoformat(row['order_timestamp']),
            last_updated=datetime.fromisoformat(row['last_updated']),
            broker_order_id=row['broker_order_id'],
            parent_order_id=row['parent_order_id'],
            tags=json.loads(row['tags']) if row['tags'] else None
        )
    
    def _log_audit(
        self, 
        operation: str, 
        entity_type: str, 
        entity_id: str, 
        old_state: Optional[Dict], 
        new_state: Optional[Dict],
        conn: sqlite3.Connection
    ):
        """Log state changes for audit trail (with retention limit)."""
        # Apply retention limit (FIXED MEDIUM #3)
        # Delete old audit logs (keep last 7 days)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        conn.execute(
            "DELETE FROM state_audit_log WHERE timestamp < ?",
            (cutoff,)
        )
        
        # Log new audit entry
        audit_id = f"AUDIT_{uuid.uuid4().hex[:8]}"
        
        old_json = json.dumps(old_state, default=str) if old_state else None
        new_json = json.dumps(new_state, default=str) if new_state else None
        
        conn.execute("""
            INSERT INTO state_audit_log (audit_id, operation, entity_type, entity_id, old_state, new_state)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (audit_id, operation, entity_type, entity_id, old_json, new_json))
    
    # ================================================================
    # CLEANUP & MAINTENANCE (FIXED CRITICAL #2)
    # ================================================================
    
    def run_maintenance(self, aggressive: bool = False, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Run database maintenance with retry logic for locked database.
        
        Args:
            aggressive: If True, run VACUUM (only in maintenance mode)
                        If False, run wal_checkpoint (safe during trading)
            max_retries: Maximum number of retry attempts if database is locked
            retry_delay: Seconds to wait between retries (doubles each attempt)
        """
        last_error = None
        delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                with self._transaction(write=True) as conn:
                    if aggressive:
                        # Only run VACUUM in maintenance mode
                        logger.warning("Running VACUUM - this will block the database")
                        conn.execute("VACUUM")
                        logger.info("VACUUM completed")
                    else:
                        # Safe checkpoint during trading
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        logger.debug("WAL checkpoint completed")
                    
                    # Clean up old data
                    self._cleanup_old_data(conn)
                    
                    # Optimize database
                    conn.execute("PRAGMA optimize")
                    return  # Success
                    
            except sqlite3.OperationalError as e:
                last_error = e
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Maintenance attempt {attempt + 1} failed (database locked), "
                                 f"retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    break
            except Exception as e:
                logger.error(f"Maintenance failed: {e}")
                return
        
        if last_error:
            logger.error(f"Maintenance failed after {max_retries} attempts: {last_error}")
    
    def _cleanup_old_data(self, conn: sqlite3.Connection):
        """Clean up old data from database."""
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=self.max_history_days)).strftime("%Y-%m-%d")
        
        # Delete old trade history
        conn.execute("""
            DELETE FROM trade_history 
            WHERE DATE(exit_time) < ?
        """, (cutoff_date,))
        
        # Delete old daily PnL records
        conn.execute("""
            DELETE FROM daily_pnl 
            WHERE date < ?
        """, (cutoff_date,))
        
        # Delete closed positions older than cutoff
        conn.execute("""
            DELETE FROM positions 
            WHERE status = 'CLOSED' 
            AND DATE(exit_time) < ?
        """, (cutoff_date,))
        
        # Delete completed orders older than cutoff
        conn.execute("""
            DELETE FROM orders 
            WHERE status IN ('COMPLETE', 'CANCELLED', 'REJECTED')
            AND DATE(order_timestamp) < ?
        """, (cutoff_date,))
        
        deleted_count = conn.total_changes
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old records")
    
    async def run_maintenance_async(self, aggressive: bool = False):
        """Async wrapper for run_maintenance."""
        return await self._run_in_executor(self.run_maintenance, aggressive)
    
    def compress_database(self) -> int:
        """
        Compress database by vacuuming (ONLY IN MAINTENANCE MODE).
        
        Returns:
            Size reduction in bytes (estimated)
        """
        # Check if we're in trading hours (basic check)
        current_hour = datetime.now(timezone.utc).hour
        if 9 <= current_hour <= 15:  # Rough NSE hours in UTC
            logger.error("Cannot run VACUUM during trading hours")
            return 0
        
        with self._transaction(write=True) as conn:
            # Get size before
            size_before = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Run checkpoint first
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            
            # Then VACUUM (FIXED CRITICAL #2)
            conn.execute("VACUUM")
            
            # Get size after
            size_after = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            reduction = size_before - size_after
            if reduction > 0:
                logger.info(f"Database compressed: {reduction} bytes saved")
            
            return reduction
    
    # ================================================================
    # DIAGNOSTICS & MONITORING
    # ================================================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._transaction() as conn:
            stats = {}
            
            # Table row counts
            tables = ['positions', 'orders', 'risk_flags', 'daily_pnl', 'trade_history', 'state_audit_log']
            for table in tables:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                stats[f'{table}_count'] = result[0]
            
            # Open positions by side
            long_result = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE side = 'LONG' AND status IN ('OPEN', 'PARTIALLY_CLOSED')"
            ).fetchone()
            short_result = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE side = 'SHORT' AND status IN ('OPEN', 'PARTIALLY_CLOSED')"
            ).fetchone()
            
            stats['open_long_positions'] = long_result[0]
            stats['open_short_positions'] = short_result[0]
            
            # Order status distribution
            order_statuses = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM orders 
                GROUP BY status
            """).fetchall()
            
            stats['order_statuses'] = {row['status']: row['count'] for row in order_statuses}
            
            # Active risk flags
            active_flags = conn.execute(
                "SELECT flag_type, COUNT(*) FROM risk_flags WHERE is_active = 1 GROUP BY flag_type"
            ).fetchall()
            
            stats['active_risk_flags'] = {row[0]: row[1] for row in active_flags}
            
            # Database size
            if self.db_path.exists():
                stats['database_size_bytes'] = self.db_path.stat().st_size
                stats['database_size_mb'] = stats['database_size_bytes'] / (1024 * 1024)
            
            # WAL file info
            wal_path = self.db_path.with_suffix('.db-wal')
            if wal_path.exists():
                stats['wal_size_bytes'] = wal_path.stat().st_size
                stats['wal_size_mb'] = stats['wal_size_bytes'] / (1024 * 1024)
            
            stats['timestamp'] = datetime.now(timezone.utc).isoformat()
            stats['schema_version'] = SCHEMA_VERSION
            
            return stats
    
    async def get_database_stats_async(self) -> Dict[str, Any]:
        """Async wrapper for get_database_stats."""
        return await self._run_in_executor(self.get_database_stats)
    
    # ================================================================
    # DECIMAL CONVERSION (FIXED MEDIUM #2)
    # ================================================================
    
    def get_daily_pnl_decimal(self) -> Tuple[Decimal, Decimal]:
        """
        Get daily PnL as Decimal for accurate risk calculations.
        
        Returns:
            Tuple of (realized_pnl_decimal, unrealized_pnl_decimal)
        """
        realized, unrealized = self.get_total_daily_pnl()
        return Decimal(str(realized)), Decimal(str(unrealized))
    
    # ================================================================
    # SHUTDOWN
    # ================================================================
    
    def close(self):
        """Close database connections and cleanup."""
        try:
            # Run final checkpoint
            conn = self.pool.get_connection()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            
            # Close connections
            self.pool.close_all()
            
            # Shutdown executor (FIXED CRITICAL #1)
            self._executor.shutdown(wait=True, cancel_futures=True)
            
            logger.info("State manager closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def close_async(self):
        """Async wrapper for close."""
        return await self._run_in_executor(self.close)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ================================================================
# FACTORY FUNCTION
# ================================================================

def create_state_manager(config_path: Optional[Path] = None) -> StateManager:
    """
    Factory function to create state manager from configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        StateManager instance
    """
    try:
        from config.config import get_settings
        
        settings = get_settings()
        db_path = settings.state_db_path
        max_history_days = 30  # Could be configurable
        
        logger.info(f"Creating state manager with database: {db_path}")
        return StateManager(db_path, max_history_days)
    
    except ImportError:
        # Fallback for testing
        logger.warning("Config module not found, using default path")
        default_path = Path("data/trading_state.db")
        default_path.parent.mkdir(parents=True, exist_ok=True)
        return StateManager(default_path)


# ================================================================
# TESTING UTILITIES
# ================================================================

if __name__ == "__main__":
    """Test the state manager."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test database
    test_db = Path("test_state.db")
    
    if test_db.exists():
        test_db.unlink()
    
    try:
        # Create state manager
        manager = StateManager(test_db)
        
        print("=" * 70)
        print("State Manager Test")
        print("=" * 70)
        
        # Test position management
        position = Position(
            position_id="POS_TEST_001",
            symbol="RELIANCE",
            side=PositionSide.LONG,
            quantity=100,
            entry_price=2500.50,
            current_price=2510.25,
            entry_time=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            status=PositionStatus.OPEN,
            unrealized_pnl=975.0
        )
        
        manager.save_position(position)
        print("✓ Position saved")
        
        # Test order management
        order = Order(
            order_id="ORD_TEST_001",
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,
            filled_quantity=0,
            price=2505.0,
            trigger_price=None,
            status=OrderStatus.PENDING,
            order_timestamp=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc)
        )
        
        manager.save_order(order)
        print("✓ Order saved")
        
        # Test risk flag
        flag_id = manager.set_risk_flag(RiskFlag.TRADING_HALTED, {"reason": "market_volatility"})
        print(f"✓ Risk flag set: {flag_id}")
        
        # Test kill switch
        print(f"✓ Is trading halted: {manager.is_trading_halted()}")
        
        # Load state
        state = manager.load_open_state()
        print(f"✓ State loaded: {len(state.positions)} positions, {len(state.open_orders)} orders")
        print(f"  Daily PnL: Realized={state.daily_realized_pnl:.2f}, "
              f"Unrealized={state.daily_unrealized_pnl:.2f}")
        
        # Get statistics
        stats = manager.get_database_stats()
        print(f"✓ Database stats: {stats['positions_count']} positions, {stats['orders_count']} orders")
        
        # Test reconciliation (FIXED CRITICAL #4)
        broker_positions = [
            {"symbol": "RELIANCE", "quantity": 100, "average_price": 2500.50},
            {"symbol": "TCS", "quantity": 50, "average_price": 3500.0},
            {"symbol": "RELIANCE", "quantity": -50, "average_price": 2500.0}  # Short position
        ]
        
        recon = manager.reconcile_with_broker(broker_positions)
        print(f"✓ Reconciliation: {recon['missing_positions']} missing, {recon['extra_positions']} extra")
        
        # Test partial close
        manager.close_position(
            "POS_TEST_001",
            exit_price=2515.0,
            exit_time=datetime.now(timezone.utc),
            realized_pnl=1450.0,
            partial_quantity=50
        )
        print("✓ Partial position closure")
        
        # Get Decimal PnL (FIXED MEDIUM #2)
        realized_dec, unrealized_dec = manager.get_daily_pnl_decimal()
        print(f"✓ Decimal PnL: Realized={realized_dec}, Unrealized={unrealized_dec}")
        
        # Test maintenance (safe checkpoint)
        manager.run_maintenance(aggressive=False)
        print("✓ Safe maintenance completed")
        
        # Cleanup
        manager.close()
        
        print("=" * 70)
        print("All tests passed!")
        print("=" * 70)
        
        # Remove test database
        test_db.unlink()
        wal_file = test_db.with_suffix('.db-wal')
        if wal_file.exists():
            wal_file.unlink()
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)