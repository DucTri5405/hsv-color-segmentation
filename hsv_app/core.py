"""Thuật toán phân đoạn ảnh theo màu trên không gian HSV (độc lập với GUI).

Pipeline (mỗi bước trả về ảnh trung gian để GUI hiển thị):
  B0  Đọc ảnh BGR uint8
  B1  Lọc Median (tùy chọn, k lẻ) để giảm nhiễu
  B2  Chuyển BGR -> HSV (H 0..179, S 0..255, V 0..255)
  B3  Tạo mask bằng cv2.inRange; nếu H_min > H_max (màu đỏ) thì OR hai dải Hue
  B4  Opening  (co rồi giãn)  -> bỏ đốm nhiễu trắng
  B5  Closing  (giãn rồi co)  -> lấp lỗ đen nhỏ
  B6  Lọc vùng nhỏ bằng connectedComponentsWithStats, đếm số vùng
      (tùy chọn) lấp khuyết viền: thay mỗi vùng bằng bao lồi của nó (vật tròn bị vệt phản sáng ở mép)
  B7  Áp mask lên ảnh gốc (bitwise_and) + overlay, đường bao, khung, số thứ tự
  B8  Đánh giá: % pixel, số vùng, thời gian; IoU/Precision/Recall/F1 nếu có mask chuẩn
"""
from dataclasses import asdict, dataclass, fields
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVConfig:
    h_min: int = 170
    h_max: int = 10
    s_min: int = 80
    s_max: int = 255
    v_min: int = 50
    v_max: int = 255
    median_size: int = 1
    kernel_size: int = 5
    open_iterations: int = 1
    close_iterations: int = 1
    min_area: int = 100
    fill_convex: int = 0

    def validate(self):
        limits = {'h_min': 179, 'h_max': 179, 's_min': 255, 's_max': 255,
                  'v_min': 255, 'v_max': 255, 'open_iterations': 10,
                  'close_iterations': 10, 'min_area': 100_000_000}
        for key, upper in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError(f'{key} phải là số nguyên từ 0 đến {upper}')
        if self.s_min > self.s_max:
            raise ValueError('S min phải nhỏ hơn hoặc bằng S max')
        if self.v_min > self.v_max:
            raise ValueError('V min phải nhỏ hơn hoặc bằng V max')
        if self.fill_convex not in (0, 1) or type(self.fill_convex) is not int:
            raise ValueError('fill_convex phải là 0 hoặc 1')
        for key in ('median_size', 'kernel_size'):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 31 or value % 2 == 0:
                raise ValueError(f'{key} phải là số lẻ từ 1 đến 31')
        return self

    @property
    def hue_wraps(self):
        return self.h_min > self.h_max

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in data.items() if k in names}).validate()

    def with_hue(self, h_min, h_max):
        d = self.to_dict(); d.update(h_min=h_min, h_max=h_max)
        return HSVConfig(**d)


# Dải Hue theo thang OpenCV (H = góc độ / 2). Đỏ nằm ở hai đầu vòng màu.
# Đỏ và Cam không chồng nhau: kẹo cam đo được H 0..17 (trung vị 6..9), kẹo đỏ H 171..179 và 0.
# Đây chỉ là điểm xuất phát; mỗi ảnh mẫu có ngưỡng riêng trong config/.
HUE_RANGES = {
    'Đỏ': (170, 5),
    'Cam': (6, 22),
    'Vàng': (22, 35),
    'Xanh lá': (35, 85),
    'Xanh dương': (90, 130),
    'Tím': (130, 160),
}
# Màu BGR để tô overlay cho từng preset
DISPLAY_BGR = {
    'Đỏ': (40, 40, 230), 'Cam': (0, 140, 255), 'Vàng': (0, 220, 255),
    'Xanh lá': (60, 200, 60), 'Xanh dương': (230, 120, 30), 'Tím': (200, 60, 170),
    'Tùy chỉnh': (0, 255, 255),
}

PRESETS = {name: HSVConfig(h_min=lo, h_max=hi) for name, (lo, hi) in HUE_RANGES.items()}
PRESETS['Tùy chỉnh'] = HSVConfig(h_min=0, h_max=179, s_min=0, v_min=0,
                                 open_iterations=0, close_iterations=0, min_area=0)


# ---------------------------------------------------------------- B2: BGR -> HSV
def bgr_to_hsv_manual(image):
    """Cài đặt công thức BGR -> HSV bằng NumPy để kiểm chứng cv2.cvtColor.

    V = max(R,G,B);  S = (V - min) / V  (S = 0 nếu V = 0)
    H (độ) = 60*(G-B)/Δ        nếu V = R
           = 120 + 60*(B-R)/Δ  nếu V = G
           = 240 + 60*(R-G)/Δ  nếu V = B      (H < 0 thì cộng 360)
    OpenCV 8-bit: H_cv = H/2 (0..179), S_cv = S*255, V_cv = V.
    """
    img = image.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    v = img.max(axis=2)
    mn = img.min(axis=2)
    delta = v - mn
    s = np.where(v > 0, delta / np.maximum(v, 1e-6) * 255.0, 0.0)
    safe = np.maximum(delta, 1e-6)
    h = np.where(v == r, 60.0 * (g - b) / safe,
                 np.where(v == g, 120.0 + 60.0 * (b - r) / safe,
                          240.0 + 60.0 * (r - g) / safe))
    h = np.where(delta == 0, 0.0, h)
    h = np.where(h < 0, h + 360.0, h)
    h_cv = np.round(h / 2.0) % 180
    return np.dstack([h_cv, np.round(s), v]).astype(np.uint8)


# ---------------------------------------------------------------- B3: ngưỡng
def hsv_threshold(hsv, config):
    """Ngưỡng hai đầu đều bao gồm. H_min > H_max: dải vòng [H_min,179] ∪ [0,H_max]."""
    c = config.validate()
    if not c.hue_wraps:
        return cv2.inRange(hsv, (c.h_min, c.s_min, c.v_min), (c.h_max, c.s_max, c.v_max))
    upper = cv2.inRange(hsv, (c.h_min, c.s_min, c.v_min), (179, c.s_max, c.v_max))
    lower = cv2.inRange(hsv, (0, c.s_min, c.v_min), (c.h_max, c.s_max, c.v_max))
    return cv2.bitwise_or(upper, lower)


def hue_band_masks(hsv, config):
    """Trả về (mask dải trên, mask dải dưới) để minh họa màu đỏ; None nếu không vòng."""
    c = config.validate()
    if not c.hue_wraps:
        return None
    upper = cv2.inRange(hsv, (c.h_min, c.s_min, c.v_min), (179, c.s_max, c.v_max))
    lower = cv2.inRange(hsv, (0, c.s_min, c.v_min), (c.h_max, c.s_max, c.v_max))
    return upper, lower


# ---------------------------------------------------------------- đánh giá
def binary_metrics(pred, gt):
    """So khớp từng pixel với mask chuẩn (0/255, cùng kích thước)."""
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape != gt.shape:
        raise ValueError('Mask chuẩn phải là ảnh 1 kênh, cùng kích thước với ảnh đang xử lý')
    p, g = pred > 0, gt > 127
    tp = int(np.count_nonzero(p & g)); fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g)); tn = int(np.count_nonzero(~p & ~g))
    empty = 1.0 if tp + fp + fn == 0 else 0.0

    def ratio(a, b):
        return a / b if b else empty
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                IoU=ratio(tp, tp + fp + fn), Precision=ratio(tp, tp + fp),
                Recall=ratio(tp, tp + fn), F1=ratio(2 * tp, 2 * tp + fp + fn))


def compare_image(pred, gt):
    """Ảnh so sánh: TP xanh lá, FP đỏ, FN xanh dương, TN đen."""
    p, g = pred > 0, gt > 127
    out = np.zeros(pred.shape + (3,), np.uint8)
    out[p & g] = (80, 200, 80); out[p & ~g] = (40, 40, 230); out[~p & g] = (230, 120, 30)
    return out


# ---------------------------------------------------------------- vẽ kết quả
def _check_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError('Ảnh đầu vào phải là ảnh màu BGR uint8 (H×W×3)')


def draw_overlay(image, layers, label=True, text_scale=1.0):
    """layers: list (mask, components, màu BGR). Tô màu mờ + đường bao + khung + số."""
    overlay = image.copy()
    for mask, _, color in layers:
        tint = np.empty_like(image); tint[:] = color
        blend = cv2.addWeighted(image, 0.45, tint, 0.55, 0)
        overlay[mask > 0] = blend[mask > 0]
    scale = max(image.shape[:2]) / 600.0
    thick = max(1, int(round(2 * scale)))
    for mask, components, color in layers:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), thick)
        if not label:
            continue
        for comp in components:
            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, thick)
            text = str(comp['id'])
            fs = 0.6 * max(scale, 0.6) * text_scale
            tt = max(thick, int(round(thick * text_scale / 1.5)))
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tt)
            ty = y - 4 if y - th - 6 >= 0 else y + th + 4
            cv2.rectangle(overlay, (x, ty - th - 4), (x + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(overlay, text, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), tt, cv2.LINE_AA)
    return overlay


def label_colors(labels, keep_ids):
    """Tô mỗi thành phần liên thông một màu (minh họa bước lọc vùng)."""
    rng = np.random.default_rng(7)
    palette = rng.integers(60, 255, size=(labels.max() + 1, 3), dtype=np.uint8)
    palette[0] = 0
    out = palette[labels]
    if keep_ids is not None:
        drop = ~np.isin(labels, list(keep_ids)) & (labels > 0)
        out[drop] = (90, 90, 90)
    return out


# ---------------------------------------------------------------- pipeline chính
def run_pipeline(image, config, display_color=None, with_steps=True, text_scale=1.0):
    """Chạy toàn bộ pipeline. Không sửa ảnh gốc."""
    _check_image(image)
    c = config.validate()
    t0 = perf_counter()
    # B1 Median
    filtered = cv2.medianBlur(image, c.median_size) if c.median_size > 1 else image
    # B2 BGR -> HSV
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    # B3 Ngưỡng
    raw = hsv_threshold(hsv, c)
    # B4, B5 Hình thái học
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.kernel_size, c.kernel_size))
    opened = (cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=c.open_iterations)
              if c.open_iterations else raw.copy())
    closed = (cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=c.close_iterations)
              if c.close_iterations else opened.copy())
    # B6 Lọc vùng nhỏ
    n, labels, stats, centers = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= c.min_area]
    lookup = np.zeros(n, np.uint8); lookup[keep] = 255
    mask = lookup[labels]
    if c.fill_convex and keep:
        mask = np.zeros_like(mask)
        for i in keep:
            pts = cv2.findNonZero((labels == i).astype(np.uint8))
            cv2.fillPoly(mask, [cv2.convexHull(pts)], 255)
    components = []
    order = sorted(keep, key=lambda i: (int(stats[i, 1]) // 40, int(stats[i, 0])))
    for number, i in enumerate(order, 1):
        x, y, w, h, area = map(int, stats[i])
        if c.fill_convex:
            area = int(np.count_nonzero(mask[y:y + h, x:x + w]))
        components.append(dict(id=number, x=x, y=y, width=w, height=h, area=area,
                               center_x=round(float(centers[i][0]), 1),
                               center_y=round(float(centers[i][1]), 1)))
    # B7 Kết quả
    segmented = cv2.bitwise_and(image, image, mask=mask)
    color = display_color or (0, 255, 255)
    overlay = draw_overlay(image, [(mask, components, color)], text_scale=text_scale)
    ms = (perf_counter() - t0) * 1000
    fg = int(np.count_nonzero(mask))
    result = dict(
        hsv=hsv, raw_mask=raw, opened=opened, closed=closed, mask=mask,
        segmented=segmented, overlay=overlay, components=components,
        stats=dict(width=image.shape[1], height=image.shape[0],
                   raw_pixels=int(np.count_nonzero(raw)),
                   raw_components=cv2.connectedComponents(raw, connectivity=8)[0] - 1,
                   opened_components=cv2.connectedComponents(opened, connectivity=8)[0] - 1,
                   opened_pixels=int(np.count_nonzero(opened)),
                   closed_pixels=int(np.count_nonzero(closed)),
                   foreground_pixels=fg,
                   foreground_percent=round(fg / mask.size * 100, 3),
                   components_before=n - 1, components=len(components),
                   processing_ms=round(ms, 2)))
    if with_steps:
        result['filtered'] = filtered
        result['labels_view'] = label_colors(labels, set(keep))
    return result



def run_multi(image, named_configs):
    """Phân đoạn nhiều màu cùng lúc: OR các mask, mỗi màu tô một màu overlay."""
    _check_image(image)
    t0 = perf_counter()
    layers, per_color = [], {}
    total = np.zeros(image.shape[:2], np.uint8)
    for name, cfg in named_configs:
        r = run_pipeline(image, cfg, with_steps=False)
        color = DISPLAY_BGR.get(name, (0, 255, 255))
        layers.append((r['mask'], r['components'], color))
        per_color[name] = dict(pixels=r['stats']['foreground_pixels'],
                               components=r['stats']['components'])
        total = cv2.bitwise_or(total, r['mask'])
    overlay = draw_overlay(image, layers, label=False)
    for mask, comps, color in layers:
        for comp in comps:
            x, y, w, h = comp['x'], comp['y'], comp['width'], comp['height']
            cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), color, 2)
    fg = int(np.count_nonzero(total))
    return dict(mask=total, segmented=cv2.bitwise_and(image, image, mask=total),
                overlay=overlay, per_color=per_color,
                stats=dict(foreground_pixels=fg, foreground_percent=round(fg / total.size * 100, 3),
                           components=sum(v['components'] for v in per_color.values()),
                           processing_ms=round((perf_counter() - t0) * 1000, 2)))


def suggest_thresholds(h, s, v, margin_h=10):
    """Gợi ý ngưỡng từ một pixel người dùng nhấp."""
    lo, hi = (h - margin_h) % 180, (h + margin_h) % 180
    return dict(h_min=int(lo), h_max=int(hi), s_min=int(max(40, s - 80)), s_max=255,
                v_min=int(max(40, v - 100)), v_max=255)
