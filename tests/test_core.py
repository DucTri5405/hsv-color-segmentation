"""Kiểm thử thuật toán: python -m pytest -q"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsv_app.core import (HSVConfig, PRESETS, bgr_to_hsv_manual, binary_metrics,  # noqa: E402
                          hsv_threshold, run_multi, run_pipeline, suggest_thresholds)
from hsv_app.io_utils import load_config, read_image, save_config, save_result  # noqa: E402
from hsv_app.samples import load_samples  # noqa: E402


def hsv_img(h, s, v, size=(40, 40)):
    return cv2.cvtColor(np.full(size + (3,), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)


def test_manual_hsv_matches_opencv_random():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    a = bgr_to_hsv_manual(img).astype(int)
    b = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    dh = np.abs(a[..., 0] - b[..., 0]); dh = np.minimum(dh, 180 - dh)  # H là vòng tròn
    assert dh.max() <= 1
    assert np.abs(a[..., 1] - b[..., 1]).max() <= 1
    assert np.abs(a[..., 2] - b[..., 2]).max() == 0


def test_manual_hsv_on_real_image():
    img = read_image(ROOT / 'data/images/01_smarties.png')
    a = bgr_to_hsv_manual(img).astype(int); b = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    dh = np.abs(a[..., 0] - b[..., 0])
    assert np.minimum(dh, 180 - dh).max() <= 1


def test_inrange_inclusive_bounds():
    hsv = np.array([[[90, 80, 50], [91, 80, 50], [89, 80, 50]]], np.uint8)
    m = hsv_threshold(hsv, HSVConfig(h_min=90, h_max=90, s_min=80, v_min=50))
    assert m.tolist() == [[255, 0, 0]]


def test_red_hue_wrap_takes_both_bands():
    hsv = np.array([[[0, 200, 200], [175, 200, 200], [90, 200, 200], [11, 200, 200]]], np.uint8)
    m = hsv_threshold(hsv, HSVConfig(h_min=170, h_max=10))
    assert m.tolist() == [[255, 255, 0, 0]]


def test_single_band_misses_wrapped_red():
    img = np.hstack([hsv_img(2, 220, 220), hsv_img(176, 220, 220)])
    one = run_pipeline(img, HSVConfig(h_min=0, h_max=10, min_area=0))
    wrap = run_pipeline(img, HSVConfig(h_min=170, h_max=10, min_area=0))
    assert one['stats']['foreground_percent'] == pytest.approx(50, abs=1)
    assert wrap['stats']['foreground_percent'] == pytest.approx(100, abs=0.1)


def test_low_saturation_needs_lower_s_min():
    img = hsv_img(110, 50, 220)
    assert run_pipeline(img, HSVConfig(h_min=95, h_max=130, s_min=80))['stats']['foreground_pixels'] == 0
    assert run_pipeline(img, HSVConfig(h_min=95, h_max=130, s_min=30, min_area=0))['stats']['foreground_percent'] == 100


def test_opening_removes_specks_closing_fills_holes():
    m = np.zeros((60, 60), np.uint8); m[10:50, 10:50] = 255; m[30, 30] = 0; m[2, 2] = 255
    img = np.zeros((60, 60, 3), np.uint8); img[m > 0] = hsv_img(60, 200, 200, (1, 1))[0, 0]
    r = run_pipeline(img, HSVConfig(h_min=35, h_max=85, kernel_size=3, min_area=0))
    assert r['raw_mask'][2, 2] == 255 and r['opened'][2, 2] == 0     # Opening bỏ đốm
    assert r['raw_mask'][30, 30] == 0 and r['closed'][30, 30] == 255  # Closing lấp lỗ


def test_min_area_filters_components():
    img = np.full((100, 100, 3), 255, np.uint8)
    blue = hsv_img(110, 220, 220, (1, 1))[0, 0]
    img[10:40, 10:40] = blue; img[70:74, 70:74] = blue
    r = run_pipeline(img, HSVConfig(h_min=95, h_max=130, open_iterations=0, close_iterations=0, min_area=50))
    assert r['stats']['components_before'] == 2 and r['stats']['components'] == 1
    assert r['components'][0]['area'] == 900


def test_pipeline_does_not_modify_input_and_keeps_colors():
    img = hsv_img(110, 220, 220, (30, 30)); copy = img.copy()
    r = run_pipeline(img, HSVConfig(h_min=95, h_max=130, min_area=0))
    assert np.array_equal(img, copy)
    assert np.array_equal(r['segmented'], img)


def test_metrics_known_values():
    pred = np.zeros((10, 10), np.uint8); gt = np.zeros((10, 10), np.uint8)
    pred[:5] = 255; gt[2:7] = 255   # TP = 30, FP = 20, FN = 20
    m = binary_metrics(pred, gt)
    assert (m['TP'], m['FP'], m['FN']) == (30, 20, 20)
    assert m['IoU'] == pytest.approx(30 / 70) and m['F1'] == pytest.approx(60 / 100)
    with pytest.raises(ValueError):
        binary_metrics(pred, gt[:5])


@pytest.mark.parametrize('bad', [dict(s_min=200, s_max=100), dict(median_size=4), dict(kernel_size=0), dict(h_min=200)])
def test_invalid_config_raises(bad):
    with pytest.raises(ValueError):
        HSVConfig(**bad).validate()


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        run_pipeline(np.zeros((5, 5), np.uint8), HSVConfig())


def test_multi_color():
    img = np.hstack([hsv_img(0, 220, 220), hsv_img(60, 220, 220), hsv_img(110, 220, 220)])
    r = run_multi(img, [(n, PRESETS[n]) for n in ('Đỏ', 'Xanh lá', 'Xanh dương')])
    assert r['stats']['foreground_percent'] == pytest.approx(100, abs=0.5)
    assert {k: v['components'] for k, v in r['per_color'].items()} == {'Đỏ': 1, 'Xanh lá': 1, 'Xanh dương': 1}


def test_suggest_thresholds_wraps():
    s = suggest_thresholds(3, 200, 200)
    assert s['h_min'] == 173 and s['h_max'] == 13


def test_config_roundtrip_and_save_result(tmp_path):
    cfg = HSVConfig(h_min=95, h_max=130, min_area=0)
    save_config(tmp_path / 'c.json', cfg)
    assert load_config(tmp_path / 'c.json') == cfg
    img = hsv_img(110, 220, 220, (20, 20))
    folder = save_result(tmp_path, img, run_pipeline(img, cfg), cfg, 'x.png')
    names = {p.name for p in folder.iterdir()}
    assert {'config.json', 'stats.json', 'components.csv', '00_anh_goc.png', '07_overlay.png'} <= names
    assert json.loads((folder / 'stats.json').read_text(encoding='utf-8'))['thong_ke']['components'] == 1


def test_all_samples_load_and_run():
    samples = load_samples(ROOT)
    real = [s for s in samples if s['group'] == 'Ảnh thật']
    assert len(real) >= 12
    for s in samples:
        img = read_image(s['image'])
        r = run_pipeline(img, s['config'])
        assert r['mask'].shape == img.shape[:2]
        if s['mask'] is not None:
            assert binary_metrics(r['mask'], read_image(s['mask'], True))['IoU'] > 0.9


def test_main_image_story():
    """Ảnh smarties, màu đỏ: đúng 4 viên kẹo."""
    s = {x['id']: x for x in load_samples(ROOT)}['01_smarties']
    r = run_pipeline(read_image(s['image']), s['config'])
    assert r['stats']['components'] == 4


def test_red_orange_presets_do_not_overlap():
    """Dải Đỏ và Cam không được trùng nhau (kẹo cam có H 0..17, kẹo đỏ H 171..179 và 0)."""
    from hsv_app.core import HUE_RANGES

    def hues(lo, hi):
        return set(range(lo, 180)) | set(range(0, hi + 1)) if lo > hi else set(range(lo, hi + 1))
    assert not hues(*HUE_RANGES['Đỏ']) & hues(*HUE_RANGES['Cam'])
    s = {x['id']: x for x in load_samples(ROOT)}['01_smarties']
    img = read_image(s['image'])
    assert run_pipeline(img, s['config'].with_hue(*HUE_RANGES['Đỏ']))['stats']['components'] == 4
    assert run_pipeline(img, s['config'].with_hue(*HUE_RANGES['Cam']))['stats']['components'] == 2


def test_fill_convex_closes_edge_notches():
    """Lấp khuyết viền: vẫn 4 kẹo, mỗi kẹo gần tròn đầy (độ đầy >= 0.985)."""
    s = {x['id']: x for x in load_samples(ROOT)}['01_smarties']
    img = read_image(s['image'])
    cfg = HSVConfig(**{**s['config'].to_dict(), 'fill_convex': 1})
    r = run_pipeline(img, cfg)
    assert r['stats']['components'] == 4
    for c in r['components']:
        m = (r['mask'][c['y']:c['y'] + c['height'], c['x']:c['x'] + c['width']] > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        area = sum(cv2.contourArea(x) for x in cnts)
        assert area / cv2.contourArea(cv2.convexHull(np.vstack(cnts))) >= 0.985
    with pytest.raises(ValueError):
        HSVConfig(fill_convex=2).validate()
