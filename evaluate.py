"""Chạy thực nghiệm trên toàn bộ dữ liệu, xuất bảng + ảnh vào outputs/.

Cách dùng:
    python evaluate.py                 # chạy thực nghiệm (dữ liệu đã có sẵn)
    python evaluate.py --tao-du-lieu   # tạo lại ảnh tổng hợp + mask chuẩn rồi chạy

Kết quả:
    outputs/anh_that_thong_ke.csv       % pixel, số vùng, thời gian trên từng ảnh thật
    outputs/so_sanh_cau_hinh.csv        IoU/F1 trên ảnh tổng hợp cho các cấu hình xử lý
    outputs/thi_nghiem_tham_so.csv      Hue 1 dải vs Hue vòng, S_min 80 vs 30, V_min...
    outputs/logo_iou.csv                IoU trên ảnh thật opencv-logo (mask chuẩn chính xác)
    outputs/ket_qua.json                toàn bộ số liệu
    outputs/luoi_truoc_sau.png          lưới ảnh trước -> sau cho các ảnh thật
    outputs/anh_that/<tên>_*.png        mask, ảnh phân đoạn, overlay của từng ảnh
    outputs/tong_hop/<tên>_*.png        mask dự đoán + ảnh so sánh TP/FP/FN
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from hsv_app.core import (DISPLAY_BGR, HSVConfig, binary_metrics, compare_image,
                          run_pipeline)
from hsv_app.io_utils import read_image, write_image
from hsv_app.samples import load_samples

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
SYN = ROOT / 'data' / 'synthetic'


# ============================================================ tạo dữ liệu
def hsv_color(h, s, v):
    """Màu BGR từ HSV (thang OpenCV)."""
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _background(rng, h=480, w=640):
    base = np.full((h, w, 3), 200, np.float32)
    base += rng.normal(0, 6, (h, w, 1))  # nền xám nhạt hơi lốm đốm
    return base


def _draw(img, mask, shape, color, target):
    kind, p = shape
    m = np.zeros(img.shape[:2], np.uint8)
    if kind == 'circle':
        cv2.circle(m, p[:2], p[2], 255, -1, cv2.LINE_8)
    elif kind == 'rect':
        cv2.rectangle(m, p[:2], p[2:], 255, -1)
    elif kind == 'ellipse':
        cv2.ellipse(m, p[:2], p[2:4], p[4], 0, 360, 255, -1)
    img[m > 0] = color
    if target:
        mask[m > 0] = 255
    else:
        mask[m > 0] = 0


def make_synthetic():
    """6 ảnh kiểm thử 640x480 có mask chuẩn, mỗi ảnh một tình huống khó."""
    rng = np.random.default_rng(2026)
    (SYN / 'images').mkdir(parents=True, exist_ok=True)
    (SYN / 'masks').mkdir(parents=True, exist_ok=True)
    shapes = [('circle', (110, 120, 60)), ('rect', (240, 60, 360, 190)), ('ellipse', (500, 120, 80, 50, 20)),
              ('circle', (120, 350, 70)), ('rect', (260, 290, 400, 420)), ('ellipse', (520, 360, 70, 90, -30))]
    specs = {}

    def new():
        return _background(rng), np.zeros((480, 640), np.uint8)

    # 1. Đỏ ở cả hai đầu vòng Hue
    img, mask = new()
    hues = [2, 176, 5, 173, 1, 178]
    for i, sh in enumerate(shapes[:4] + shapes[5:]):
        _draw(img, mask, sh, hsv_color(hues[i], 220, 210), True)
    _draw(img, mask, shapes[4], hsv_color(16, 220, 230), False)  # cam gây nhiễu
    cv2.circle(img, (330, 240), 35, hsv_color(145, 200, 200), -1)  # tím gây nhiễu
    specs['syn_01_do_hue_vong'] = (img, mask, 'Đỏ', 'Đỏ nằm ở hai đầu vòng Hue (H≈1–5 và 173–178)',
                                   dict(h_min=170, h_max=10, s_min=80, v_min=50))

    # 2. Xanh dương nhạt (S thấp)
    img, mask = new()
    for i, sh in enumerate(shapes[:5]):
        _draw(img, mask, sh, hsv_color(110, 45 + 8 * i, 220), True)
    _draw(img, mask, shapes[5], hsv_color(60, 160, 200), False)
    specs['syn_02_xanh_nhat'] = (img, mask, 'Xanh dương', 'Xanh dương nhạt, S chỉ 45–77',
                                 dict(h_min=95, h_max=130, s_min=30, v_min=50))

    # 3. Nhiễu xung (muối tiêu màu) 6%
    img, mask = new()
    for sh in shapes:
        _draw(img, mask, sh, hsv_color(60, 200, 190), True)
    noise = rng.random((480, 640)) < 0.15
    img[noise] = rng.choice([0, 255], size=(int(noise.sum()), 3))
    specs['syn_03_nhieu_muoi_tieu'] = (img, mask, 'Xanh lá', 'Xanh lá + 15% nhiễu xung màu',
                                       dict(h_min=35, h_max=85, s_min=80, v_min=50))

    # 4. Ánh sáng không đều + bóng
    img, mask = new()
    for sh in shapes:
        _draw(img, mask, sh, hsv_color(28, 210, 250), True)
    gx = np.linspace(0.3, 1.0, 640)[None, :, None]
    shadow = np.ones((480, 640), np.float32)
    cv2.ellipse(shadow, (250, 330), (210, 120), 0, 0, 360, 0.55, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 25)[..., None]
    img = img * gx * shadow
    specs['syn_04_bong_toi'] = (img, mask, 'Vàng', 'Vàng dưới ánh sáng giảm dần + vùng bóng',
                                dict(h_min=20, h_max=35, s_min=80, v_min=40))

    # 5. Nhiễu Gauss mạnh
    img, mask = new()
    for sh in shapes:
        _draw(img, mask, sh, hsv_color(115, 170, 170), True)
    img = img + rng.normal(0, 40, img.shape)
    specs['syn_05_nhieu_gauss'] = (img, mask, 'Xanh dương', 'Xanh dương + nhiễu Gauss σ=40',
                                   dict(h_min=95, h_max=130, s_min=80, v_min=50))

    # 6. Các màu gần nhau: cam cạnh đỏ và vàng
    img, mask = new()
    cols = [(16, True), (3, False), (29, False), (16, True), (16, True), (4, False)]
    for sh, (hh, tgt) in zip(shapes, cols):
        _draw(img, mask, sh, hsv_color(hh, 220, 230), tgt)
    specs['syn_06_mau_gan_nhau'] = (img, mask, 'Cam', 'Cam đặt cạnh đỏ và vàng (Hue gần nhau)',
                                    dict(h_min=10, h_max=22, s_min=80, v_min=50))

    for name, (img, mask, target, note, thr) in specs.items():
        img = np.clip(img, 0, 255).astype(np.uint8)
        write_image(SYN / 'images' / f'{name}.png', img)
        write_image(SYN / 'masks' / f'{name}.png', mask)
        cfg = dict(thr, s_max=255, v_max=255, median_size=5, kernel_size=3,
                   open_iterations=1, close_iterations=1, min_area=150)
        data = dict(anh=f'data/synthetic/images/{name}.png', mau_muc_tieu=target, ghi_chu=note,
                    nhom='Ảnh tổng hợp', mask_chuan=f'data/synthetic/masks/{name}.png', config=cfg)
        (ROOT / 'config' / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                      encoding='utf-8')
    print('Đã tạo', len(specs), 'ảnh tổng hợp')


LOGO_COLORS = {'do': (68, 42, 255), 'xanh_la': (103, 218, 139), 'xanh_duong': (255, 141, 18)}


def make_logo_masks():
    """Mask chuẩn opencv-logo lấy từ kênh alpha + màu tô thuần (không dùng HSV)."""
    src = ROOT / 'data' / 'images' / '08_opencv_logo.png'
    masks = ROOT / 'data' / 'masks'
    raw = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if raw.ndim != 3 or raw.shape[2] != 4:
        print('opencv-logo đã được ghép nền trắng, giữ nguyên mask chuẩn hiện có')
        return
    alpha = raw[..., 3]
    colors = raw[..., :3].astype(np.int32)
    names = list(LOGO_COLORS)
    refs = [LOGO_COLORS[n] for n in names] + [(0, 0, 0)]  # thêm màu đen của tam giác giữa logo
    nearest = np.stack([np.abs(colors - np.array(c)).sum(2) for c in refs]).argmin(0)
    for i, n in enumerate(names):
        write_image(masks / f'08_opencv_logo_{n}.png', ((alpha > 127) & (nearest == i)).astype(np.uint8) * 255)
    # Ghép lên nền trắng, lưu lại 3 kênh để mọi phần mềm hiển thị giống nhau
    a = alpha[..., None].astype(np.float32) / 255
    write_image(src, (raw[..., :3] * a + 255 * (1 - a)).astype(np.uint8))
    print('Đã tạo mask chuẩn opencv-logo')


def make_video():
    """Video 6 giây từ smarties: xoay, dịch chuyển, độ sáng thay đổi (demo chức năng video)."""
    src = read_image(ROOT / 'data' / 'images' / '01_smarties.png')
    out = ROOT / 'data' / 'videos' / 'smarties_chuyen_dong.avi'
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H, fps, n = 640, 480, 25, 150
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    for i in range(n):
        t = i / n
        angle = 360 * t
        scale = 1.0 + 0.15 * np.sin(2 * np.pi * t)
        cx = W / 2 + 90 * np.sin(2 * np.pi * t); cy = H / 2 + 40 * np.cos(2 * np.pi * t)
        M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), angle, scale)
        M[0, 2] += cx - src.shape[1] / 2; M[1, 2] += cy - src.shape[0] / 2
        frame = cv2.warpAffine(src, M, (W, H), borderValue=(255, 255, 255))
        light = 0.75 + 0.25 * np.cos(2 * np.pi * t)  # ánh sáng thay đổi theo thời gian
        writer.write(np.clip(frame.astype(np.float32) * light, 0, 255).astype(np.uint8))
    writer.release()
    print('Đã tạo video', out.name)


# ============================================================ thực nghiệm
def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def fit_h(img, h):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def real_images(samples):
    rows, tiles = [], []
    for s in samples:
        if s['group'] != 'Ảnh thật':
            continue
        img = read_image(s['image'])
        r = run_pipeline(img, s['config'], DISPLAY_BGR.get(s['target']))
        st = r['stats']
        # thời gian xử lý: trung vị 5 lần chạy, chỉ tính thuật toán (không vẽ ảnh minh họa)
        st['processing_ms'] = round(float(np.median(
            [run_pipeline(img, s['config'], with_steps=False)['stats']['processing_ms'] for _ in range(5)])), 2)
        d = OUT / 'anh_that'
        for key in ('mask', 'segmented', 'overlay'):
            write_image(d / f"{s['id']}_{key}.png", r[key])
        rows.append(dict(anh=s['image'].name, mau=s['target'], rong=st['width'], cao=st['height'],
                         pixel_chon=st['foreground_pixels'], phan_tram=st['foreground_percent'],
                         vung_truoc_loc=st['components_before'], vung_sau_loc=st['components'],
                         thoi_gian_ms=st['processing_ms'], ghi_chu=s['note']))
        tiles.append(np.hstack([fit_h(img, 240), np.full((240, 8, 3), 255, np.uint8),
                                fit_h(r['segmented'], 240)]))
    save_csv(OUT / 'anh_that_thong_ke.csv', rows)
    # Lưới trước -> sau, 2 cột
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 6, 6, 6, 6 + w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for t in tiles]
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    write_image(OUT / 'luoi_truoc_sau.png', grid)
    return rows


PROCESSING = [
    ('HSV thuần', dict(median_size=1, open_iterations=0, close_iterations=0, min_area=0)),
    ('+ Opening/Closing k=3', dict(median_size=1, kernel_size=3, open_iterations=1, close_iterations=1, min_area=0)),
    ('+ Opening/Closing k=7', dict(median_size=1, kernel_size=7, open_iterations=1, close_iterations=1, min_area=0)),
    ('+ Median 5 + O/C k=3', dict(median_size=5, kernel_size=3, open_iterations=1, close_iterations=1, min_area=0)),
    ('Đầy đủ (+ lọc vùng 150)', dict(median_size=5, kernel_size=3, open_iterations=1, close_iterations=1, min_area=150)),
]


def synthetic_experiments(samples):
    syn = [s for s in samples if s['group'] == 'Ảnh tổng hợp']
    per_image, summary = [], []
    for label, change in PROCESSING:
        ious, f1s = [], []
        for s in syn:
            img, gt = read_image(s['image']), read_image(s['mask'], True)
            cfg = replace(s['config'], **change)
            r = run_pipeline(img, cfg, with_steps=False)
            m = binary_metrics(r['mask'], gt)
            ious.append(m['IoU']); f1s.append(m['F1'])
            per_image.append(dict(cau_hinh=label, anh=s['id'], IoU=round(m['IoU'], 4), F1=round(m['F1'], 4),
                                  Precision=round(m['Precision'], 4), Recall=round(m['Recall'], 4)))
            if label.startswith('Đầy đủ') or label == 'HSV thuần':
                tag = 'day_du' if label.startswith('Đầy đủ') else 'hsv_thuan'
                write_image(OUT / 'tong_hop' / f"{s['id']}_{tag}_mask.png", r['mask'])
                write_image(OUT / 'tong_hop' / f"{s['id']}_{tag}_so_sanh.png", compare_image(r['mask'], gt))
        summary.append(dict(cau_hinh=label, IoU_tb=round(float(np.mean(ious)), 4),
                            F1_tb=round(float(np.mean(f1s)), 4), so_anh=len(syn)))
    save_csv(OUT / 'so_sanh_cau_hinh.csv', summary)
    save_csv(OUT / 'so_sanh_cau_hinh_tung_anh.csv', per_image)
    return summary, per_image


def parameter_experiments(samples):
    by_id = {s['id']: s for s in samples}
    rows = []

    def run(sid, label, **change):
        s = by_id[sid]
        img, gt = read_image(s['image']), read_image(s['mask'], True)
        r = run_pipeline(img, replace(s['config'], **change), with_steps=False)
        m = binary_metrics(r['mask'], gt)
        rows.append(dict(thi_nghiem=label, anh=sid, IoU=round(m['IoU'], 4), F1=round(m['F1'], 4),
                         Recall=round(m['Recall'], 4), pixel_chon=r['stats']['foreground_pixels']))
        return r

    for lab, lo, hi in [('Hue 1 dải [0,10]', 0, 10), ('Hue 1 dải [170,179]', 170, 179),
                        ('Hue vòng [170,179]∪[0,10]', 170, 10)]:
        r = run('syn_01_do_hue_vong', lab, h_min=lo, h_max=hi)
        write_image(OUT / 'tham_so' / f'hue_{lo}_{hi}.png', r['mask'])
    for smin in (80, 30):
        r = run('syn_02_xanh_nhat', f'S_min = {smin}', s_min=smin)
        write_image(OUT / 'tham_so' / f's_min_{smin}.png', r['mask'])
    for vmin in (120, 40):
        r = run('syn_04_bong_toi', f'V_min = {vmin}', v_min=vmin)
        write_image(OUT / 'tham_so' / f'v_min_{vmin}.png', r['mask'])
    for med in (1, 5):
        r = run('syn_03_nhieu_muoi_tieu', f'Median {med}' if med > 1 else 'Không Median', median_size=med)
        write_image(OUT / 'tham_so' / f'median_{med}.png', r['mask'])
    save_csv(OUT / 'thi_nghiem_tham_so.csv', rows)
    return rows


def logo_experiment():
    img = read_image(ROOT / 'data' / 'images' / '08_opencv_logo.png')
    rows = []
    for key, name in [('do', 'Đỏ'), ('xanh_la', 'Xanh lá'), ('xanh_duong', 'Xanh dương')]:
        gt = read_image(ROOT / 'data' / 'masks' / f'08_opencv_logo_{key}.png', True)
        lo, hi = {'Đỏ': (170, 10), 'Xanh lá': (35, 85), 'Xanh dương': (90, 130)}[name]
        cfg = HSVConfig(h_min=lo, h_max=hi, s_min=100, v_min=100, kernel_size=3, min_area=100)
        r = run_pipeline(img, cfg, with_steps=False)
        m = binary_metrics(r['mask'], gt)
        rows.append(dict(mau=name, IoU=round(m['IoU'], 4), F1=round(m['F1'], 4),
                         Precision=round(m['Precision'], 4), Recall=round(m['Recall'], 4)))
    save_csv(OUT / 'logo_iou.csv', rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tao-du-lieu', action='store_true', help='tạo lại ảnh tổng hợp và mask chuẩn')
    args = ap.parse_args()
    if args.tao_du_lieu or not (SYN / 'images').exists():
        make_synthetic(); make_logo_masks(); make_video()
    OUT.mkdir(exist_ok=True)
    samples = load_samples(ROOT)
    real = real_images(samples)
    summary, per_image = synthetic_experiments(samples)
    params = parameter_experiments(samples)
    logo = logo_experiment()
    report = dict(anh_that=real, so_sanh_cau_hinh=summary, so_sanh_tung_anh=per_image,
                  thi_nghiem_tham_so=params, logo=logo)
    (OUT / 'ket_qua.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Ảnh thật: {len(real)} | ảnh tổng hợp: {summary[0]["so_anh"]}')
    for row in summary:
        print(f"  {row['cau_hinh']:<28} IoU={row['IoU_tb']:.4f}  F1={row['F1_tb']:.4f}")
    for row in params:
        print(f"  {row['thi_nghiem']:<28} {row['anh']:<24} IoU={row['IoU']:.4f}")
    for row in logo:
        print(f"  logo {row['mau']:<10} IoU={row['IoU']:.4f}  F1={row['F1']:.4f}")
    print('Đã ghi kết quả vào', OUT)


if __name__ == '__main__':
    main()
