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
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config.config import get_settings
from core.engine import TradingEngine, handle_exit_signals

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
    logger.info(f"  Startup Time (UTC): {datetime.utcnow().isoformat()}")
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
    """Main async entry point with fail-safe resource cleanup."""
    setup_logging()
    logger = logging.getLogger("Main")
    settings = get_settings()
    
    logger.info("THE ALGO MACHINE - Initializing...")
    
    log_system_fingerprint(logger, settings)
    
    is_valid, missing_keys = validate_environment()
    if not is_valid:
        logger.error(f"FATAL: Missing mandatory configuration: {', '.join(missing_keys)}")
        sys.exit(1)
        
    engine = None
    try:
        # Initialize Engine
        engine = TradingEngine()
        
        # 1. Defensive Signal Handling
        try:
            handle_exit_signals(engine)
        except (NotImplementedError, RuntimeError):
            logger.warning("OS signals not supported. Relying on KeyboardInterrupt.")
        
        # 2. Main Run Flow
        logger.info("System Ready. Starting core loops...")
        await engine.run()
        
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Shutting down...")
    except Exception as e:
        logger.critical(f"UNHANDLED CRITICAL EXCEPTION: {e}", exc_info=True)
    finally:
        # 3. Force Shutdown Path
        if engine:
            logger.info("Ensuring engine resources are cleaned up...")
            await engine.shutdown()
        logger.info("The Algo Machine - Process Terminated.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
