"""Danh sách ảnh mẫu + preset JSON trong thư mục config/."""
import json
import sys
from pathlib import Path

from .core import HSVConfig


def app_root():
    """Thư mục gốc: cạnh file .exe khi đóng gói, hoặc thư mục project khi chạy Python."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = app_root()


def load_samples(root=ROOT):
    """Đọc mọi file config/*.json có khóa 'anh' -> danh sách mẫu (theo tên file)."""
    items = []
    for f in sorted((root / 'config').glob('*.json')):
        try:
            raw = json.loads(f.read_text(encoding='utf-8'))
            image = root / raw['anh']
            if not image.exists():
                continue
            mask = root / raw['mask_chuan'] if raw.get('mask_chuan') else None
            items.append(dict(id=f.stem, image=image, target=raw.get('mau_muc_tieu', 'Tùy chỉnh'),
                              note=raw.get('ghi_chu', ''), mask=mask if mask and mask.exists() else None,
                              config=HSVConfig.from_dict(raw['config']), group=raw.get('nhom', 'Ảnh thật')))
        except (KeyError, ValueError, json.JSONDecodeError, TypeError):
            continue
    return items
