import logging
import os
import sys
import locale
from datetime import datetime

def setup_logger(name='crop_mapping', log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    if sys.platform == 'win32':
        console_encoding = locale.getpreferredencoding()
        if console_encoding != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                try:
                    import win32console
                    win32console.SetConsoleOutputCP(65001)
                except ImportError:
                    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{current_time}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"System platform: {sys.platform}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Default encoding: {sys.getdefaultencoding()}")
    logger.info(f"Console encoding: {locale.getpreferredencoding()}")
    logger.info(f"File system encoding: {sys.getfilesystemencoding()}")
    
    return logger
