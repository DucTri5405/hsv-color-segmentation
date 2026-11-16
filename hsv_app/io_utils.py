"""Đọc/ghi ảnh (hỗ trợ đường dẫn tiếng Việt trên Windows), cấu hình JSON, gói kết quả."""
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .core import HSVConfig, binary_metrics, compare_image

IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def read_image(path, grayscale=False):
    p = Path(path)
    try:
        data = np.frombuffer(p.read_bytes(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f'Không đọc được file: {p.name}') from exc
    if image is None:
        raise ValueError(f'File không phải ảnh hợp lệ: {p.name}')
    return image


def write_image(path, image):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(p.suffix or '.png', image)
    if not ok:
        raise OSError(f'Không mã hóa được ảnh {p.name}')
    p.write_bytes(data.tobytes())


def load_config(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict) and 'config' in raw:
        raw = raw['config']
    return HSVConfig.from_dict(raw)


def save_config(path, config, extra=None):
    data = dict(extra or {}); data['config'] = config.validate().to_dict()
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def new_output_dir(parent, stem='ket_qua'):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    i = 1
    while folder.exists():
        folder = parent / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"; i += 1
    folder.mkdir()
    return folder


def hue_view(hsv):
    """Kênh H (0..179) giãn ra 0..255 để hiển thị."""
    return (hsv[..., 0].astype(np.float32) * 255 / 179).astype(np.uint8)


def step_images(original, result):
    """Danh sách (tên file, ảnh) theo đúng thứ tự pipeline."""
    hsv = result['hsv']
    return [('00_anh_goc', original), ('01_median', result.get('filtered', original)),
            ('02_kenh_H', hue_view(hsv)), ('02_kenh_S', hsv[..., 1]), ('02_kenh_V', hsv[..., 2]),
            ('03_mask_tho_inRange', result['raw_mask']), ('04_sau_opening', result['opened']),
            ('05_sau_closing', result['closed']), ('06_sau_loc_vung', result['mask']),
            ('07_anh_phan_doan', result['segmented']), ('07_overlay', result['overlay'])]


def save_result(parent, original, result, config, source='', gt=None, frame_index=None):
    folder = new_output_dir(parent, 'ket_qua')
    for name, image in step_images(original, result):
        write_image(folder / f'{name}.png', image)
    evaluation = None
    if gt is not None:
        evaluation = binary_metrics(result['mask'], gt)
        write_image(folder / '08_so_sanh_mask_chuan.png', compare_image(result['mask'], gt))
    (folder / 'config.json').write_text(json.dumps(dict(nguon=str(source), frame=frame_index,
                                                        config=config.to_dict()),
                                                   ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'stats.json').write_text(json.dumps(dict(thong_ke=result['stats'], danh_gia=evaluation),
                                                  ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'components.csv').open('w', newline='', encoding='utf-8-sig') as f:
        keys = ['id', 'x', 'y', 'width', 'height', 'area', 'center_x', 'center_y']
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows(result['components'])
    return folder
