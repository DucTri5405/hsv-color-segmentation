"""Chạy giao diện HSV Color Segmentation.

    python main.py                      # mở app, nạp sẵn ảnh smarties
    python main.py --sample 02_peppers  # nạp ảnh mẫu khác
    python main.py --video data/videos/smarties_chuyen_dong.avi
"""
from hsv_app.gui import main

if __name__ == '__main__':
    main()
