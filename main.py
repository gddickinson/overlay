#!/usr/bin/env python3
"""
Main entry point for the Advanced TIFF Stack Viewer application.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg

from utils.logger import setup_logger
from utils.config import load_config, save_config
from gui.main_window import MainWindow


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced TIFF Stack Viewer')

    parser.add_argument('--fluorescence', '-f', type=str, help='Path to fluorescence TIFF stack')
    parser.add_argument('--mask', '-m', type=str, help='Path to mask TIFF stack')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', '-c', type=str, help='Path to configuration file')

    return parser.parse_args()


def main():
    """Application entry point."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    debug_mode = True  # Force debug mode on
    logger = setup_logger(debug_mode)
    logger.info("Starting Advanced TIFF Stack Viewer")

    # Load configuration
    config_path = args.config if args.config else None
    config = load_config(config_path)

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced TIFF Stack Viewer")
    app.setOrganizationName("MicroscopyLab")

    # Set pyqtgraph configuration
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)

    # Create main window
    window = MainWindow(config, logger)
    window.show()

    # Load files from command line if provided
    if args.fluorescence:
        window.load_fluorescence_stack(args.fluorescence)
    if args.mask:
        window.load_mask_stack(args.mask)

    # Start event loop
    exit_code = app.exec()

    # Save configuration on exit
    save_config(config)

    # Return exit code
    logger.info(f"Application exiting with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
