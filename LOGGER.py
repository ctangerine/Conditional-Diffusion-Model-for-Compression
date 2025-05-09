import logging # Thêm import logging
import sys   # Thêm import sys
import os

import logging
import sys
import os

try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False # Sẽ fallback về logger không màu nếu colorlog chưa cài

# --- Logger Setup ---
def setup_logger(name='my_app', log_file=None, level=logging.DEBUG):
    """Thiết lập logger với tên, file log (tùy chọn) và cấp độ, có màu cho console."""
    logger_instance = logging.getLogger(name)
    
    # Chỉ cấu hình nếu logger chưa có handlers, tránh nhân đôi khi gọi lại hoặc khi có nhiều module cùng gọi
    if not logger_instance.handlers: # Giữ nguyên logic kiểm tra handlers của bạn
        logger_instance.setLevel(level)
        
        # --- Console Handler (với màu sắc nếu có colorlog) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if COLORLOG_AVAILABLE:
            log_colors_config = {
                'DEBUG':    'cyan',       # Xanh lơ cho debug - nhẹ nhàng
                'INFO':     'blue',      # Xanh lá cho info - thông tin chung
                'WARNING':  'bold_yellow',     # Vàng cho warning - cảnh báo
                'ERROR':    'red',        # Đỏ cho error - lỗi
                'CRITICAL': 'bold_purple',   # Đỏ đậm/đậm hơn cho critical - lỗi nghiêm trọng
            }
            # Định dạng chuỗi log, %(log_color)s sẽ được thay thế bằng mã màu tương ứng
            console_formatter = ColoredFormatter(
                fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S', # Định dạng thời gian nếu muốn
                reset=True, # Reset màu sau mỗi message để không ảnh hưởng tới các output khác
                log_colors=log_colors_config,
                secondary_log_colors={}, # Có thể dùng để tô màu các phần khác của message
                style='%' # Kiểu định dạng (%, { hoặc $)
            )
        else:
            # Định dạng console mặc định nếu colorlog không có
            # (giống như formatter của file)
            print("Thư viện colorlog chưa được cài đặt, log trên console sẽ không có màu. "
                  "Bạn có thể cài đặt bằng: pip install colorlog", file=sys.stderr)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S')
        
        console_handler.setFormatter(console_formatter)
        logger_instance.addHandler(console_handler)

        # --- File Handler (luôn không màu) ---
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Giữ nguyên mode='w' (ghi đè) như trong code bạn cung cấp
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8') 
            file_handler.setLevel(level)
            # Định dạng cho file (không màu)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)-8s - %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            logger_instance.addHandler(file_handler)
        
        # Ngăn logger lan truyền message lên logger cha (root logger)
        logger_instance.propagate = False # Giữ nguyên thiết lập của bạn
        
    return logger_instance