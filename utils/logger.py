import logging
import os
import sys
import locale
from datetime import datetime

def setup_logger(name='crop_mapping', log_dir='logs'):
    """
    Set up logger.
    Args:
        name (str): Logger name
        log_dir (str): Directory to save log files
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # If handlers already exist, do not add again
    if logger.handlers:
        return logger
    
    # Set console encoding
    if sys.platform == 'win32':
        # Check console encoding
        console_encoding = locale.getpreferredencoding()
        if console_encoding != 'utf-8':
            # Try setting console encoding to UTF-8
            try:
                # For Windows 10 and above
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                # For older Windows versions, use win32console
                try:
                    import win32console
                    win32console.SetConsoleOutputCP(65001)  # 65001 is the code page for UTF-8
                except ImportError:
                    # If win32console cannot be imported, use environment variable
                    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)  # Use stdout instead of stderr
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')  # Specify utf-8 encoding
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log encoding information
    logger.info(f"System platform: {sys.platform}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Default encoding: {sys.getdefaultencoding()}")
    logger.info(f"Console encoding: {locale.getpreferredencoding()}")
    logger.info(f"File system encoding: {sys.getfilesystemencoding()}")
    
    return logger 