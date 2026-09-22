"""
core/logging_setup.py - Centralized UTF-8 safe logging configuration for Windows/Linux.
"""
import os
import sys
import logging
import logging.handlers

def force_utf8_env():
    """
    On Windows, forces stdout and stderr to reconfigure to utf-8 if available,
    preventing console output crashes on non-ASCII characters.
    """
    if sys.platform == 'win32':
        # Reconfigure sys.stdout and sys.stderr to utf-8 if supported
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass
        if hasattr(sys.stderr, 'reconfigure'):
            try:
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                pass

def setup_logging():
    """
    Sets up a centralized, non-duplicating, rotating logging structure with UTF-8 encoding.
    Configures loggers for:
      - 'main' -> logs/main.log
      - 'swarm' -> logs/orchestrator.log
      - 'thermo' -> logs/thermo.log
      - 'providers' -> logs/providers.log
      - Root logger -> logs/runtime.log
    """
    force_utf8_env()
    os.makedirs('logs', exist_ok=True)

    # Prevent duplicating handlers if setup_logging is called multiple times
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s')

    # Handler for central runtime execution errors/exceptions (logs/runtime.log)
    runtime_handler = logging.handlers.RotatingFileHandler(
        'logs/runtime.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    runtime_handler.setFormatter(formatter)
    root_logger.addHandler(runtime_handler)

    # Console Handler (UTF-8 safe)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Set up specific loggers (they inherit console and runtime output from root but write specific info to separate files)
    loggers_config = {
        'main': 'logs/main.log',
        'swarm': 'logs/orchestrator.log',
        'thermo': 'logs/thermo.log',
        'providers': 'logs/providers.log',
    }

    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True  # Inherit console and root handler

        # Rotating File Handler with explicit UTF-8 encoding
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger('memory').propagate = True
    logging.getLogger('trainer').propagate = True

    # Initial log statement
    logging.getLogger('main').info("UTF-8 Logging initialized successfully.")
