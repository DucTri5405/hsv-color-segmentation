"""Xuất video kết quả theo từng frame (AVI MJPG, không âm thanh)."""
import json
import math

import cv2

from .core import run_pipeline
from .io_utils import new_output_dir


def export_video(source, parent, config, progress=None, cancel=None, display_color=None):
    config.validate()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Không mở được video (file lỗi hoặc thiếu codec)')
    writers = []; count = 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Video không có frame đọc được')
        h, w = frame.shape[:2]
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        folder = new_output_dir(parent, 'video')
        for name in ('goc', 'mask', 'overlay'):
            writer = cv2.VideoWriter(str(folder / f'{name}.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
            if not writer.isOpened():
                raise OSError('Không tạo được file video AVI')
            writers.append(writer)
        stopped = False
        while ok:
            if cancel is not None and cancel.is_set():
                stopped = True; break
            r = run_pipeline(frame, config, display_color, with_steps=False)
            for writer, item in zip(writers, (frame, cv2.cvtColor(r['mask'], cv2.COLOR_GRAY2BGR), r['overlay'])):
                writer.write(item)
            count += 1
            if progress:
                progress(count, total)
            ok, frame = cap.read()
        report = dict(nguon=str(source), config=config.to_dict(), so_frame=count, fps=fps,
                      rong=w, cao=h, trang_thai='đã dừng' if stopped else 'hoàn tất')
        (folder / 'video.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return folder, report
    finally:
        cap.release()
        for writer in writers:
            writer.release()
