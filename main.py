# main.py
"""
Entry point for The Algo Machine.
Initializes the system, sets up logging, and starts the Trading Engine.
Harden version: Protects against duplicate logging, unsafe signals, and fatal crashes.
"""

import asyncio
import logging
import sys
import os
import platform
import signal
import atexit
import subprocess
import threading
import time
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path

# Fix for UnicodeEncodeError on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from config.config import get_settings
from core.engine import TradingEngine, handle_exit_signals

# Global reference to dashboard process for cleanup
_dashboard_process = None


def start_dashboard():
    """
    Start the dashboard as a background process.
    Returns the process object or None if failed.
    """
    global _dashboard_process
    
    # Get logger (setup_logging must be called first)
    logger = logging.getLogger("Main")
    
    try:
        # Get dashboard script path
        dashboard_script = Path(__file__).parent / "dashboard" / "dashboard_api.py"
        
        if not dashboard_script.exists():
            logger.warning(f"Dashboard script not found: {dashboard_script}")
            return None
        
        # Start dashboard process
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        creationflags = 0
        preexec_fn = None

        if sys.platform == 'win32':
            # Windows: Use CREATE_NEW_PROCESS_GROUP to allow clean termination
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Linux/Mac
            preexec_fn = os.setpgrp

        _dashboard_process = subprocess.Popen(
            [sys.executable, str(dashboard_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, # Capture stderr for debugging if needed
            text=True,
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn
        )
        
        logger.info(f"Dashboard started with PID {_dashboard_process.pid}")
        return _dashboard_process
        
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return None

def stop_dashboard():
    """Stop the dashboard process if running."""
    global _dashboard_process
    
    # Get logger
    logger = logging.getLogger("Main")
    
    if _dashboard_process is None:
        return
    
    try:
        logger.info("Stopping dashboard process...")
        
        if _dashboard_process.poll() is None:  # Process still running
            # Try graceful termination first
            _dashboard_process.terminate()
            
            try:
                # Wait up to 2 seconds for graceful shutdown
                _dashboard_process.wait(timeout=2)
                logger.info("Dashboard stopped gracefully.")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning("Dashboard not responding, forcing termination...")
                _dashboard_process.kill()
                _dashboard_process.wait()
                logger.info("Dashboard force-terminated.")
        else:
            logger.info("Dashboard already stopped.")
            
    except Exception as e:
        # Use print if logger might be closed
        print(f"Error stopping dashboard: {e}")
    finally:
        _dashboard_process = None

def cleanup_on_exit():
    """
    Cleanup function called on program exit.
    Ensures all resources are released.
    """
    stop_dashboard()
    
    # Force close any remaining asyncio tasks
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
    except:
        pass

# Register cleanup handler
atexit.register(cleanup_on_exit)

def setup_logging():
    """Configures structured logging for console and file with duplication protection."""
    root_logger = logging.getLogger()
    
    # 1. Guard against duplicate handlers (fix for restarts/reloads)
    if root_logger.hasHandlers():
        return
        
    # Ensure logs directory exists
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "logs"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Root level configured to INFO; components can use DEBUG if needed
    root_logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Hierarchical Control: Silence verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Enable DEBUG for specific high-frequency components if requested by config
    # logging.getLogger("alpha").setLevel(logging.DEBUG)

def log_system_fingerprint(logger, settings):
    """Logs system metadata and strategy configuration for post-mortems."""
    logger.info("="*60)
    logger.info("SYSTEM PROFILE")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {sys.version.split(' ')[0]}")
    logger.info(f"  Startup Time (UTC): {datetime.now(timezone.utc).isoformat()}")
    logger.info("="*60)
    
    logger.info("STRATEGY FINGERPRINT")
    logger.info(f"  Max Positions: {settings.max_positions}")
    logger.info(f"  Max Daily Loss: {settings.max_daily_loss}")
    logger.info(f"  Max Consecutive Losses: {settings.max_consecutive_losses}")
    logger.info(f"  Alpha Entry Threshold: {settings.entry_alpha_threshold}")
    logger.info(f"  Pegged Orders: {settings.use_pegged_orders}")
    
    # Securely confirm credentials
    if settings.broker_api_key and settings.broker_api_key != "your_api_key":
        logger.info(f"  Broker API Key: {settings.broker_api_key[:4]}...{settings.broker_api_key[-4:]}")
    if settings.broker_access_token and settings.broker_access_token != "your_access_token":
        logger.info("  Broker Access Token: VALIDATED")
    logger.info("="*60)

def validate_environment():
    """Ensure mandatory environment variables/settings are present."""
    settings = get_settings()
    required = {
        "Broker API Key": settings.broker_api_key,
        "Broker Access Token": settings.broker_access_token,
    }
    
    missing = [name for name, val in required.items() 
              if not val or val in ["your_api_key", "your_access_token"]]
    
    if missing:
        return False, missing
    return True, []

async def main():
    """Main async entry point with proper resource cleanup."""
    setup_logging()
    logger = logging.getLogger("Main")
    settings = get_settings()
    
    # 2. Formal Settings Contract Check
    # Fail fast if critical configuration fields are missing (🛡️ Safety Improvement)
    REQUIRED_SETTINGS = [
        "signal_cooldown_sec",
        "alpha_poll_interval_ms",
        "market_data_stale_threshold_sec",
        "entry_alpha_threshold",
        "max_daily_loss",
        "max_positions",
        "max_single_order_value",
        "execution_tick_size",
        "order_placement_delay_ms",
        "execution_order_poll_interval_ms",
        "dry_run_mode",
        "trading_mode"
    ]
    
    missing_settings = [k for k in REQUIRED_SETTINGS if not hasattr(settings, k)]
    if missing_settings:
        logger.critical(f"FATAL: Missing critical configuration fields: {', '.join(missing_settings)}")
        raise RuntimeError(f"Settings Contract Violation: Missing {missing_settings}")
    
    logger.info("Settings contract validated successfully.")
    logger.info("THE ALGO MACHINE - Initializing...")
    
    log_system_fingerprint(logger, settings)
    
    is_valid, missing_keys = validate_environment()
    if not is_valid:
        logger.error(f"FATAL: Missing mandatory configuration: {', '.join(missing_keys)}")
        sys.exit(1)
    
    # Start dashboard (if available)
    start_dashboard()
    
    # Wait a moment for dashboard to start
    await asyncio.sleep(1)
    logger.info("Dashboard is running in the background.")
    
    engine = None
    try:
        # Initialize Engine
        engine = TradingEngine()
        
        # 1. Setup signal handlers (defensive - may not work on Windows)
        try:
            handle_exit_signals(engine)
        except (NotImplementedError, RuntimeError):
            logger.warning("OS signals not supported. Relying on KeyboardInterrupt.")
        
        # 2. Main Run Flow
        logger.info("System Ready. Starting core loops...")
        await engine.run()
        
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down...")
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down...")
    except Exception as e:
        logger.critical(f"UNHANDLED CRITICAL EXCEPTION: {e}", exc_info=True)
    finally:
        # 3. Force Shutdown Path
        if engine:
            logger.info("Ensuring engine resources are cleaned up...")
            try:
                await asyncio.wait_for(engine.shutdown(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Engine shutdown timeout - forcing cleanup")
            except Exception as e:
                logger.error(f"Error during engine shutdown: {e}")
        
        # 4. Stop dashboard
        stop_dashboard()
        
        logger.info("The Algo Machine - Process Terminated.")

if __name__ == "__main__":
    # Setup signal handler for Ctrl+C
    def signal_handler(signum, frame):
        """Handle Ctrl+C by raising an error that main() catches.

        Raising ensures the code falls into the `finally` block of
        `asyncio.run(main())` so `engine.shutdown()` is awaited.
        """
        print("\n🛑 Interrupt received, initiating engine shutdown...")
        # Raising this ensures we hit the 'finally' block in main()
        raise KeyboardInterrupt
    
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run main coroutine
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Shutdown complete.")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        # Final cleanup
        cleanup_on_exit()
        
        # Allow normal shutdown: wait briefly for threads to exit
        time.sleep(1.0)
        
        # Watchdog: Only force-exit if critical threads are still hanging
        # This prevents state corruption while ensuring process termination
        if threading.active_count() > 1:
            print("⚠️  Forcing exit due to lingering threads:")
            for thread in threading.enumerate():
                if not thread.daemon and thread is not threading.main_thread():
                    try:
                        print(f"   - {thread.name}")
                    except Exception:
                        pass
            os._exit(0)
