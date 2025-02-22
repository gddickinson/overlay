"""
Logging configuration for the application.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
import traceback
from datetime import datetime


class LogHandler:
    """Handler for application logs with console and file outputs."""
    
    def __init__(self, debug=False):
        self.logger = logging.getLogger('tiff_stack_viewer')
        self.debug = debug
        self.setup_logger()
        
        # Register global exception handler
        sys.excepthook = self.handle_exception
    
    def setup_logger(self):
        """Set up logger with both console and file handlers."""
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Set logging level based on debug mode
        level = logging.DEBUG if self.debug else logging.INFO
        self.logger.setLevel(level)
        
        # Create formatters
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        log_dir = self._get_log_directory()
        log_file = log_dir / f"tsv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Debug mode: {self.debug}")
        self.logger.info(f"Log file: {log_file}")
    
    def _get_log_directory(self):
        """Create and return the log directory."""
        if sys.platform == 'win32':
            base_dir = os.path.expandvars('%LOCALAPPDATA%')
        else:
            base_dir = os.path.expanduser('~')
        
        log_dir = Path(base_dir) / '.tiff_stack_viewer' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log the exception
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        self.logger.critical(f"Unhandled exception:\n{tb_text}")
        
        # Also print to stderr
        print(f"An unexpected error occurred: {exc_value}", file=sys.stderr)


def setup_logger(debug=False):
    """Initialize and return the application logger."""
    handler = LogHandler(debug)
    return handler.logger


class LogCapture:
    """Context manager to capture logs during a specific operation."""
    
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.log_records = []
    
    def __enter__(self):
        self.logger.info(f"Starting: {self.operation_name}")
        self.handler = LogCaptureHandler(self.log_records)
        self.logger.addHandler(self.handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
        if exc_type:
            self.logger.error(f"Error in {self.operation_name}: {exc_val}")
            return False
        self.logger.info(f"Completed: {self.operation_name}")
        return True
    
    def get_logs(self):
        """Return captured log messages."""
        return [record.getMessage() for record in self.log_records]


class LogCaptureHandler(logging.Handler):
    """Handler to capture log records in a list."""
    
    def __init__(self, records):
        super().__init__()
        self.records = records
    
    def emit(self, record):
        self.records.append(record)
