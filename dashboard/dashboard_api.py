#!/usr/bin/env python3
"""
Antigravity Trading Dashboard API Server
READ-ONLY: Serves trading data to the dashboard without modifying anything.

This server:
- Reads from SQLite database (READ-ONLY mode)
- Parses log files (READ-ONLY)
- Serves data via REST API
- NO WRITE OPERATIONS
- NO TRADE INTERFERENCE
"""

from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for dashboard access

# Configuration
# Resolve paths relative to this script file to ensure they work 
# regardless of CWD (whether run from root or dashboard folder)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "trading_state.db"
LOGS_DIR = PROJECT_ROOT / "logs"

# ================================================================
# READ-ONLY DATABASE FUNCTIONS
# ================================================================

def get_db_connection(read_only=True):
    """
    Get SQLite database connection in READ-ONLY mode.
    This ensures we never accidentally write to the database.
    """
    uri = f"file:{DB_PATH}?mode=ro" if read_only else DB_PATH
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def dict_from_row(row):
    """Convert sqlite3.Row to dict."""
    return dict(row) if row else None

# ================================================================
# API ENDPOINTS
# ================================================================

@app.route('/api/state', methods=['GET'])
def get_state():
    """
    Get current trading state including positions, orders, and P&L.
    READ-ONLY: No modifications made.
    """
    try:
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        
        # Get open positions
        positions_rows = cursor.execute("""
            SELECT position_id, symbol, side, quantity, entry_price, 
                   current_price, entry_time, last_updated, status,
                   exit_price, exit_time, realized_pnl, unrealized_pnl, tags
            FROM positions
            WHERE status = 'OPEN'
            ORDER BY entry_time DESC
        """).fetchall()
        
        positions = [dict_from_row(row) for row in positions_rows]
        
        # Get recent orders (last 50)
        orders_rows = cursor.execute("""
            SELECT order_id, symbol, side, quantity, filled_quantity, 
                   price, trigger_price, status, order_timestamp, 
                   last_updated, broker_order_id, tags
            FROM orders
            ORDER BY order_timestamp DESC
            LIMIT 50
        """).fetchall()
        
        orders = [dict_from_row(row) for row in orders_rows]
        
        # Get daily P&L metrics
        today = datetime.now(timezone.utc).date().isoformat()
        
        pnl_row = cursor.execute("""
            SELECT SUM(realized_pnl) as total_realized,
                   SUM(unrealized_pnl) as total_unrealized
            FROM pnl_records
            WHERE date = ?
        """, (today,)).fetchone()
        
        daily_realized_pnl = pnl_row['total_realized'] if pnl_row['total_realized'] else 0.0
        daily_unrealized_pnl = pnl_row['total_unrealized'] if pnl_row['total_unrealized'] else 0.0
        
        # If no PnL records, calculate from positions
        if daily_realized_pnl == 0 and daily_unrealized_pnl == 0:
            for pos in positions:
                daily_unrealized_pnl += pos.get('unrealized_pnl', 0.0)
        
        # Get trades today
        trades_today = cursor.execute("""
            SELECT COUNT(*) as count
            FROM trade_history
            WHERE DATE(exit_time) = DATE('now')
        """).fetchone()['count']
        
        # Get risk flags
        risk_flags_rows = cursor.execute("""
            SELECT flag_type, flag_data, timestamp
            FROM risk_flags
            WHERE is_active = 1
        """).fetchall()
        
        risk_flags = [row['flag_type'] for row in risk_flags_rows]
        
        # Check config for dry_run mode (if config table exists)
        try:
            config_row = cursor.execute("""
                SELECT value FROM config WHERE key = 'dry_run_mode'
            """).fetchone()
            dry_run_mode = config_row['value'] == 'True' if config_row else True
        except:
            # Default to True for safety
            dry_run_mode = True
        
        conn.close()
        
        return jsonify({
            'positions': positions,
            'open_orders': [o for o in orders if o['status'] in ['OPEN', 'PENDING', 'PARTIAL_FILL']],
            'orders': orders,
            'daily_realized_pnl': daily_realized_pnl,
            'daily_unrealized_pnl': daily_unrealized_pnl,
            'trades_today': trades_today,
            'risk_flags': risk_flags,
            'dry_run_mode': dry_run_mode,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """
    Get all positions.
    READ-ONLY: No modifications made.
    """
    try:
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        
        rows = cursor.execute("""
            SELECT position_id, symbol, side, quantity, entry_price, 
                   current_price, entry_time, last_updated, status,
                   exit_price, exit_time, realized_pnl, unrealized_pnl, tags
            FROM positions
            ORDER BY entry_time DESC
            LIMIT 100
        """).fetchall()
        
        positions = [dict_from_row(row) for row in rows]
        conn.close()
        
        return jsonify(positions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """
    Get recent orders.
    READ-ONLY: No modifications made.
    """
    try:
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        
        rows = cursor.execute("""
            SELECT order_id, symbol, side, quantity, filled_quantity, 
                   price, trigger_price, status, order_timestamp, 
                   last_updated, broker_order_id, tags
            FROM orders
            ORDER BY order_timestamp DESC
            LIMIT 100
        """).fetchall()
        
        orders = [dict_from_row(row) for row in rows]
        conn.close()
        
        return jsonify(orders)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """
    Get trade history.
    READ-ONLY: No modifications made.
    """
    try:
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        
        rows = cursor.execute("""
            SELECT trade_id, position_id, symbol, side, quantity,
                   entry_price, exit_price, entry_time, exit_time,
                   realized_pnl, holding_period_seconds, tags
            FROM trade_history
            ORDER BY exit_time DESC
            LIMIT 100
        """).fetchall()
        
        trades = [dict_from_row(row) for row in rows]
        conn.close()
        
        return jsonify(trades)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """
    Get recent log entries.
    READ-ONLY: Parses log files without modification.
    """
    try:
        logs = []
        logs_path = Path(LOGS_DIR)
        
        if not logs_path.exists():
            return jsonify([])
        
        # Get most recent log file
        log_files = sorted(logs_path.glob('trading_*.log'), key=os.path.getmtime, reverse=True)
        
        if not log_files:
            return jsonify([])
        
        recent_log = log_files[0]
        
        # Parse log file (last 200 lines)
        with open(recent_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-200:]
        
        # Log format: 2024-02-02 10:30:45,123 - ModuleName - LEVEL - [Thread] - Message
        log_pattern = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - ([\w\.]+) - (\w+) - \[.*?\] - (.+)'
        )
        
        for line in lines:
            match = log_pattern.match(line.strip())
            if match:
                timestamp, module, level, message = match.groups()
                logs.append({
                    'timestamp': timestamp,
                    'module': module,
                    'level': level,
                    'message': message
                })
            else:
                # Include unparsed lines as-is (might be multiline exceptions)
                if line.strip():
                    logs.append({
                        'timestamp': '',
                        'module': '',
                        'level': 'RAW',
                        'message': line.strip()
                    })
        
        return jsonify(logs)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get aggregated metrics and statistics.
    READ-ONLY: No modifications made.
    """
    try:
        conn = get_db_connection(read_only=True)
        cursor = conn.cursor()
        
        # Daily stats
        today = datetime.now(timezone.utc).date().isoformat()
        
        trades_today = cursor.execute("""
            SELECT COUNT(*) as count, 
                   SUM(realized_pnl) as total_pnl,
                   AVG(realized_pnl) as avg_pnl,
                   AVG(holding_period_seconds) as avg_hold_time
            FROM trade_history
            WHERE DATE(exit_time) = DATE('now')
        """).fetchone()
        
        # Open positions stats
        positions_stats = cursor.execute("""
            SELECT COUNT(*) as count,
                   SUM(unrealized_pnl) as total_unrealized
            FROM positions
            WHERE status = 'OPEN'
        """).fetchone()
        
        # Win rate (all time)
        win_rate_stats = cursor.execute("""
            SELECT 
                COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losses,
                COUNT(*) as total
            FROM trade_history
        """).fetchone()
        
        win_rate = (win_rate_stats['wins'] / win_rate_stats['total'] * 100) if win_rate_stats['total'] > 0 else 0
        
        conn.close()
        
        return jsonify({
            'trades_today': trades_today['count'] or 0,
            'daily_pnl': trades_today['total_pnl'] or 0.0,
            'avg_pnl_per_trade': trades_today['avg_pnl'] or 0.0,
            'avg_hold_time_seconds': trades_today['avg_hold_time'] or 0,
            'open_positions_count': positions_stats['count'] or 0,
            'unrealized_pnl': positions_stats['total_unrealized'] or 0.0,
            'win_rate': round(win_rate, 2),
            'total_trades': win_rate_stats['total'] or 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'Antigravity Dashboard API',
        'mode': 'READ-ONLY',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("="*60)
    print("⚡ ANTIGRAVITY TRADING DASHBOARD API SERVER ⚡")
    print("="*60)
    print(f"Database: {DB_PATH} (READ-ONLY)")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Server: http://localhost:5000")
    print(f"Dashboard: Open antigravity_dashboard.html in browser")
    print("="*60)
    print("⚠️  READ-ONLY MODE: No database writes, No trade interference ⚠️")
    print("="*60)
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"\n⚠️  WARNING: Database file '{DB_PATH}' not found!")
        print("Make sure the trading system is in the same directory,")
        print("or update DB_PATH in this script.\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
