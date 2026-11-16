"""Giao diện Tkinter tiếng Việt cho HSV Segmentation (chữ to, rõ khi chiếu máy chiếu)."""
import argparse
import math
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .core import (DISPLAY_BGR, HUE_RANGES, PRESETS, HSVConfig, binary_metrics,
                   compare_image, run_multi, run_pipeline, suggest_thresholds)
from .io_utils import IMAGE_TYPES, hue_view, load_config, read_image, save_config, save_result
from .samples import ROOT, load_samples
from .video import export_video

BG = '#eef2f7'
DARK = '#13213a'
ACCENT = '#0b4f9c'
FONT = ('Segoe UI', 12)
FONT_B = ('Segoe UI', 12, 'bold')
FONT_H = ('Segoe UI', 13, 'bold')


# ------------------------------------------------------------------ ô hiển thị ảnh
class ImagePanel(tk.Frame):
    def __init__(self, parent, title, on_pick=None):
        super().__init__(parent, bg=BG)
        self.title = tk.Label(self, text=title, font=FONT_B, bg=BG, fg=DARK, anchor='w')
        self.title.pack(fill='x', padx=4, pady=(2, 0))
        self.canvas = tk.Canvas(self, bg='#1b2638', highlightthickness=0, width=200, height=140)
        self.canvas.pack(fill='both', expand=True, padx=3, pady=(0, 3))
        self.array = None; self.photo = None; self.mapping = None; self.on_pick = on_pick
        self.canvas.bind('<Configure>', lambda e: self.draw())
        if on_pick:
            self.canvas.bind('<Button-1>', self.pick)
            self.canvas.configure(cursor='crosshair')

    def show(self, array, title=None):
        if title is not None:
            self.title.configure(text=title)
        self.array = array; self.draw()

    def clear(self, text=''):
        self.array = None; self.placeholder = text; self.draw()

    def draw(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.array is None:
            self.canvas.delete('all')
            if getattr(self, 'placeholder', ''):
                self.canvas.create_text(cw // 2, ch // 2, text=self.placeholder, fill='#c9d3e3', font=FONT, width=240)
            return
        if cw < 4 or ch < 4:
            return
        h, w = self.array.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        if self.array.ndim == 3:
            rgb = cv2.cvtColor(self.array, cv2.COLOR_BGR2RGB)
            method = Image.Resampling.BILINEAR
        else:
            rgb = self.array; method = Image.Resampling.NEAREST
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((nw, nh), method))
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor='center')
        self.mapping = ((cw - nw) // 2, (ch - nh) // 2, nw, nh, w, h)

    def pick(self, event):
        if self.mapping is None:
            return
        ox, oy, nw, nh, w, h = self.mapping
        if ox <= event.x < ox + nw and oy <= event.y < oy + nh:
            self.on_pick(min(w - 1, int((event.x - ox) * w / nw)), min(h - 1, int((event.y - oy) * h / nh)))


# ------------------------------------------------------------------ thanh màu HSV
class HSVRangeBar(tk.Canvas):
    """Minh họa dải H đang chọn trên vòng Hue và bảng màu (H × S) được giữ lại."""
    W, H = 340, 150

    def __init__(self, parent):
        super().__init__(parent, width=self.W, height=self.H, bg=BG, highlightthickness=0)
        self.photo = None

    def update_range(self, c):
        w = self.W - 20
        hue = np.linspace(0, 179, w).astype(np.uint8)
        # Hàng 1: toàn bộ thang Hue 0..179
        strip = np.dstack([np.tile(hue, (24, 1)), np.full((24, w), 255, np.uint8), np.full((24, w), 255, np.uint8)])
        strip = cv2.cvtColor(strip, cv2.COLOR_HSV2RGB)
        inside = (hue >= c.h_min) & (hue <= c.h_max) if not c.hue_wraps else (hue >= c.h_min) | (hue <= c.h_max)
        strip[:, ~inside] = (strip[:, ~inside] * 0.22 + 40).astype(np.uint8)
        # Hàng 2: các màu được giữ (trục ngang = H trong dải, trục dọc = S từ cao xuống thấp)
        hs = (np.arange(c.h_min, c.h_min + ((c.h_max - c.h_min) % 180) + 1) % 180).astype(np.uint8)
        if hs.size == 0:
            hs = np.array([c.h_min], np.uint8)
        xs = hs[np.linspace(0, hs.size - 1, w).astype(int)]
        ss = np.linspace(c.s_max, c.s_min, 40).astype(np.uint8)
        patch = np.dstack([np.tile(xs, (40, 1)), np.tile(ss[:, None], (1, w)),
                           np.full((40, w), max(c.v_max, 1), np.uint8)])
        patch = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)
        vs = np.linspace(c.v_min, c.v_max, w).astype(np.uint8)
        mid_h = int((c.h_min + ((c.h_max - c.h_min) % 180) / 2) % 180)
        vstrip = np.dstack([np.full((16, w), mid_h, np.uint8), np.full((16, w), (c.s_min + c.s_max) // 2, np.uint8),
                            np.tile(vs, (16, 1))])
        vstrip = cv2.cvtColor(vstrip, cv2.COLOR_HSV2RGB)
        img = np.full((self.H, self.W, 3), 238, np.uint8)
        img[0:24, 10:10 + w] = strip
        img[46:86, 10:10 + w] = patch
        img[110:126, 10:10 + w] = vstrip
        self.photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.delete('all')
        self.create_image(0, 0, image=self.photo, anchor='nw')
        for v in (0, 30, 60, 90, 120, 150, 179):
            x = 10 + v / 179 * (w - 1)
            self.create_text(x, 34, text=str(v), font=('Segoe UI', 9), fill=DARK)
        self.create_text(10, 97, anchor='w', font=('Segoe UI', 9), fill=DARK,
                         text=f'Màu giữ lại: H {c.h_min}→{c.h_max} (ngang), S {c.s_max}→{c.s_min} (dọc)')
        self.create_text(10, 138, anchor='w', font=('Segoe UI', 9), fill=DARK,
                         text=f'Độ sáng V từ {c.v_min} đến {c.v_max}')


# ------------------------------------------------------------------ ứng dụng
PIPE_TITLES = ['Ảnh gốc (nhấp lấy màu)', 'B1. Lọc Median', 'B2. Kênh H (0–179)', 'B2. Kênh S (0–255)',
               'B2. Kênh V (0–255)', 'B3. Mask thô (inRange)', 'B4. Sau Opening', 'B5. Sau Closing',
               'B6. Lọc vùng nhỏ', 'B7. Ảnh phân đoạn', 'B7. Overlay + số vùng', 'So với mask chuẩn']


class HSVApplication:
    def __init__(self, root):
        self.root = root
        root.title('HSV Color Segmentation – Phân đoạn ảnh theo màu')
        root.configure(bg=BG)
        try:
            root.state('zoomed')
        except tk.TclError:
            root.geometry('1366x768')
        root.minsize(1100, 680)
        self.image = None; self.result = None; self.result_config = None; self.gt = None
        self.multi = None; self.source = ''; self.cap = None; self.playing = False
        self.frame_index = None; self.pending = None; self.video_pending = None; self.changing = False
        self.export_thread = None; self.events = queue.Queue(); self.cancel_export = threading.Event()
        self.closing = False; self.delay = 40; self.suggestion = None
        self.samples = load_samples()
        self.sample_map = {f"{s['id']} • {s['target']}": s for s in self.samples}
        self._style()
        self.build()
        self.set_config(PRESETS['Xanh dương'])
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Control-o>', lambda e: self.open_image())
        root.bind('<Control-s>', lambda e: self.save())
        root.after(100, self.poll_events)
        root.report_callback_exception = self.report_error

    # ---------------------------------------------------------- dựng giao diện
    def _style(self):
        st = ttk.Style(); st.theme_use('clam')
        st.configure('.', font=FONT, background=BG)
        st.configure('TFrame', background=BG)
        st.configure('TLabel', background=BG, foreground=DARK)
        st.configure('TButton', padding=(10, 6), font=FONT)
        st.configure('Accent.TButton', font=FONT_B, foreground='white', background=ACCENT)
        st.map('Accent.TButton', background=[('active', '#1565c0')])
        st.configure('TNotebook.Tab', font=FONT_H, padding=(18, 6), background='#c9d4e4', foreground=DARK)
        st.map('TNotebook.Tab', background=[('selected', ACCENT)], foreground=[('selected', 'white')])
        st.configure('TCheckbutton', background=BG, font=FONT)
        st.configure('TCombobox', padding=4)
        self.root.option_add('*TCombobox*Listbox.font', FONT)

    def build(self):
        bar = tk.Frame(self.root, bg=ACCENT); bar.pack(fill='x')
        tk.Label(bar, text='PHÂN ĐOẠN ẢNH THEO MÀU – KHÔNG GIAN HSV', font=('Segoe UI', 17, 'bold'),
                 bg=ACCENT, fg='white').pack(side='left', padx=14, pady=6)
        tk.Label(bar, text='HSV Color Segmentation', font=FONT, bg=ACCENT, fg='#dce8f8').pack(side='right', padx=14)
        tools = ttk.Frame(self.root, padding=(8, 6)); tools.pack(fill='x')
        self.play_text = tk.StringVar(value='▶ Phát')
        for text, cmd in [('📂 Mở ảnh', self.open_image), ('🎬 Mở video', self.open_video),
                          (None, self.toggle_play), ('🎯 Nạp mask chuẩn', self.load_gt),
                          ('💾 Lưu kết quả', self.save), ('Lưu cấu hình', self.save_settings),
                          ('Nạp cấu hình', self.load_settings), ('Xuất video', self.export)]:
            b = ttk.Button(tools, text=text, command=cmd) if text else ttk.Button(tools, textvariable=self.play_text, command=cmd)
            b.pack(side='left', padx=2)
        ttk.Button(tools, text='❓ Hướng dẫn', command=self.help).pack(side='right', padx=2)

        body = ttk.Frame(self.root, padding=(8, 0)); body.pack(fill='both', expand=True)
        left = ttk.Frame(body, width=372); left.pack(side='left', fill='y'); left.pack_propagate(False)
        sc = tk.Canvas(left, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient='vertical', command=sc.yview)
        sc.configure(yscrollcommand=sb.set); sb.pack(side='right', fill='y'); sc.pack(fill='both', expand=True)
        ctl = ttk.Frame(sc, padding=(4, 4, 8, 4)); win = sc.create_window((0, 0), window=ctl, anchor='nw')
        ctl.bind('<Configure>', lambda e: sc.configure(scrollregion=sc.bbox('all')))
        sc.bind('<Configure>', lambda e: sc.itemconfigure(win, width=e.width))
        self._scroll_canvas, self._left = sc, left
        self.root.bind_all('<MouseWheel>', self._wheel, add='+')

        def head(text):
            tk.Label(ctl, text=text, font=FONT_H, bg=BG, fg=ACCENT, anchor='w').pack(fill='x', pady=(8, 2))

        head('Ảnh mẫu (data/images)')
        self.sample_var = tk.StringVar()
        cb = ttk.Combobox(ctl, textvariable=self.sample_var, values=list(self.sample_map), state='readonly', font=FONT)
        cb.pack(fill='x'); cb.bind('<<ComboboxSelected>>', lambda e: self.load_sample())
        self.sample_note = tk.Label(ctl, text='', font=('Segoe UI', 10), bg=BG, fg='#44546e', wraplength=340, justify='left', anchor='w')
        self.sample_note.pack(fill='x')

        head('Màu cần tách (preset)')
        self.preset = tk.StringVar(value='Xanh dương')
        pb = ttk.Combobox(ctl, textvariable=self.preset, values=list(PRESETS), state='readonly', font=FONT)
        pb.pack(fill='x'); pb.bind('<<ComboboxSelected>>', lambda e: self.apply_preset())
        self.range_bar = HSVRangeBar(ctl); self.range_bar.pack(pady=(6, 0))

        head('Ngưỡng H, S, V')
        self.vars = {}
        grid = ttk.Frame(ctl); grid.pack(fill='x')
        for row, (key, label, upper) in enumerate([('h_min', 'H min', 179), ('h_max', 'H max', 179),
                                                   ('s_min', 'S min', 255), ('s_max', 'S max', 255),
                                                   ('v_min', 'V min', 255), ('v_max', 'V max', 255)]):
            v = tk.IntVar(); self.vars[key] = v
            tk.Label(grid, text=label, font=FONT_B, bg=BG, fg=DARK, width=6, anchor='w').grid(row=row, column=0, sticky='s')
            tk.Scale(grid, from_=0, to=upper, orient='horizontal', variable=v, font=('Segoe UI', 10),
                     bg=BG, highlightthickness=0, troughcolor='#c8d3e3', length=270, sliderlength=22,
                     command=lambda _v: self.schedule()).grid(row=row, column=1, sticky='ew')
        grid.columnconfigure(1, weight=1)
        tk.Label(ctl, text='H min > H max ⇒ lấy dải vòng [H min,179] ∪ [0,H max] (màu đỏ)',
                 font=('Segoe UI', 10), bg=BG, fg='#8a1c1c', wraplength=340, justify='left').pack(fill='x')

        head('Hậu xử lý')
        for key, label, values in [('median_size', 'Median (1 = tắt)', [1, 3, 5, 7, 9]),
                                   ('kernel_size', 'Kernel elip', [1, 3, 5, 7, 9, 11, 15]),
                                   ('open_iterations', 'Số lần Opening', list(range(6))),
                                   ('close_iterations', 'Số lần Closing', list(range(6)))]:
            line = ttk.Frame(ctl); line.pack(fill='x', pady=2)
            ttk.Label(line, text=label).pack(side='left')
            var = tk.IntVar(); self.vars[key] = var
            box = ttk.Combobox(line, values=values, textvariable=var, width=5, state='readonly', font=FONT)
            box.pack(side='right'); box.bind('<<ComboboxSelected>>', lambda e: self.schedule())
        line = ttk.Frame(ctl); line.pack(fill='x', pady=2)
        ttk.Label(line, text='Diện tích tối thiểu (px)').pack(side='left')
        self.vars['min_area'] = tk.IntVar(value=0)
        sp = ttk.Spinbox(line, from_=0, to=100000, increment=50, textvariable=self.vars['min_area'], width=7,
                         font=FONT, command=self.schedule)
        sp.pack(side='right'); sp.bind('<KeyRelease>', lambda e: self.schedule())

        head('Nhiều màu cùng lúc (tab Trước/Sau)')
        mf = ttk.Frame(ctl); mf.pack(fill='x')
        self.multi_vars = {}
        for i, name in enumerate(HUE_RANGES):
            v = tk.BooleanVar(value=False); self.multi_vars[name] = v
            ttk.Checkbutton(mf, text=name, variable=v, command=self.schedule).grid(row=i // 3, column=i % 3, sticky='w', padx=2)
        tk.Label(ctl, text='Dùng chung S, V, hậu xử lý ở trên; mỗi màu tô một màu riêng.',
                 font=('Segoe UI', 10), bg=BG, fg='#44546e', wraplength=340, justify='left').pack(fill='x')

        head('Lấy màu từ ảnh')
        self.pick_text = tk.StringVar(value='Nhấp vào ảnh gốc để xem giá trị H, S, V của pixel.')
        tk.Label(ctl, textvariable=self.pick_text, font=FONT, bg=BG, fg=DARK, wraplength=340, justify='left', anchor='w').pack(fill='x')
        self.swatch = tk.Label(ctl, text='', bg=BG, width=4, height=1, relief='groove'); self.swatch.pack(anchor='w', pady=2)
        ttk.Button(ctl, text='Dùng ngưỡng gợi ý', command=self.use_suggestion).pack(fill='x', pady=(2, 10))

        right = ttk.Frame(body); right.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.tabs = ttk.Notebook(right); self.tabs.pack(fill='both', expand=True)
        t1 = ttk.Frame(self.tabs); t2 = ttk.Frame(self.tabs)
        self.tabs.add(t1, text='  Trước / Sau  '); self.tabs.add(t2, text='  Pipeline từng bước  ')
        self.tabs.bind('<<NotebookTabChanged>>', lambda e: self.render())
        self.panels = []
        for i, title in enumerate(['Ảnh gốc (nhấp để lấy màu)', 'Mask nhị phân', 'Ảnh phân đoạn (giữ màu gốc)', 'Overlay + số vùng']):
            p = ImagePanel(t1, title, self.pick_color if i == 0 else None)
            p.grid(row=i // 2, column=i % 2, sticky='nsew'); self.panels.append(p)
        for j in range(2):
            t1.columnconfigure(j, weight=1); t1.rowconfigure(j, weight=1)
        self.pipe_panels = []
        for i, title in enumerate(PIPE_TITLES):
            p = ImagePanel(t2, title, self.pick_color if i == 0 else None)
            p.grid(row=i // 4, column=i % 4, sticky='nsew'); self.pipe_panels.append(p)
        for j in range(4):
            t2.columnconfigure(j, weight=1)
        for j in range(3):
            t2.rowconfigure(j, weight=1)

        info = tk.Frame(self.root, bg='#dfe7f2'); info.pack(fill='x', side='bottom')
        self.stats_text = tk.StringVar(value='Chưa có ảnh. Chọn ảnh mẫu hoặc bấm Mở ảnh.')
        self.metric_text = tk.StringVar(value='')
        self.status = tk.StringVar(value='Sẵn sàng.')
        tk.Label(info, textvariable=self.stats_text, font=('Segoe UI', 13, 'bold'), bg='#dfe7f2', fg=DARK, anchor='w').pack(fill='x', padx=12, pady=(5, 0))
        tk.Label(info, textvariable=self.metric_text, font=('Segoe UI', 13, 'bold'), bg='#dfe7f2', fg='#0a6b2f', anchor='w').pack(fill='x', padx=12)
        tk.Label(info, textvariable=self.status, font=FONT, bg='#dfe7f2', fg='#44546e', anchor='w').pack(fill='x', padx=12, pady=(0, 2))
        self.progress = ttk.Progressbar(info, mode='determinate'); self.progress.pack(fill='x', padx=12, pady=(0, 5))

    def _wheel(self, event):
        w = self.root.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w == self._left:
                self._scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units'); return
            w = getattr(w, 'master', None)

    # ---------------------------------------------------------- tham số
    def config(self):
        return HSVConfig(**{k: int(v.get()) for k, v in self.vars.items()}).validate()

    def set_config(self, config):
        self.changing = True
        for k, v in config.to_dict().items():
            self.vars[k].set(v)
        self.changing = False
        self.schedule()

    def apply_preset(self):
        name = self.preset.get()
        if name == 'Tùy chỉnh':
            self.set_config(PRESETS[name]); return
        try:
            cur = self.config()
        except (ValueError, tk.TclError):
            cur = PRESETS[name]
        lo, hi = HUE_RANGES[name]
        self.set_config(cur.with_hue(lo, hi))

    def schedule(self, *_):
        if self.changing:
            return
        if self.pending:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(60, self.process)

    # ---------------------------------------------------------- xử lý
    def display_color(self):
        return DISPLAY_BGR.get(self.preset.get(), (0, 255, 255))

    def process(self):
        self.pending = None
        try:
            c = self.config()
        except (ValueError, tk.TclError) as exc:
            self.status.set('Tham số chưa hợp lệ: ' + str(exc)); return
        self.range_bar.update_range(c)
        if self.image is None:
            return
        try:
            self.result = run_pipeline(self.image, c, self.display_color())
            self.result_config = c
            chosen = [n for n, v in self.multi_vars.items() if v.get()]
            self.multi = run_multi(self.image, [(n, c.with_hue(*HUE_RANGES[n])) for n in chosen]) if chosen else None
        except (ValueError, cv2.error) as exc:
            self.status.set('Không xử lý được: ' + str(exc)); return
        self.render()
        t = self.result['stats']
        if self.multi:
            parts = ', '.join(f"{n}: {d['components']} vùng" for n, d in self.multi['per_color'].items())
            self.stats_text.set(f"Nhiều màu – {parts} | Tổng {self.multi['stats']['foreground_percent']:.2f}% ảnh")
        else:
            self.stats_text.set(f"Pixel được chọn: {t['foreground_percent']:.2f}%   |   Số vùng: {t['components']} "
                                f"(trước lọc {t['components_before']})   |   Thời gian: {t['processing_ms']:.1f} ms   |   "
                                f"Ảnh {t['width']}×{t['height']}")
        if self.gt is not None:
            if self.gt.shape == self.result['mask'].shape:
                m = binary_metrics(self.result['mask'], self.gt)
                self.metric_text.set(f"So với mask chuẩn:  IoU = {m['IoU']:.4f}   F1 = {m['F1']:.4f}   "
                                     f"Precision = {m['Precision']:.4f}   Recall = {m['Recall']:.4f}")
            else:
                self.metric_text.set('Mask chuẩn khác kích thước ảnh – bỏ qua đánh giá.')
        else:
            self.metric_text.set('IoU / F1: chưa nạp mask chuẩn (ảnh tổng hợp và opencv-logo có sẵn mask).')
        frame = f' • frame {self.frame_index}' if self.cap is not None else ''
        self.status.set(f'Nguồn: {Path(self.source).name}{frame}')

    def render(self):
        if self.result is None or self.image is None:
            return
        r = self.result
        if self.tabs.index('current') == 0:
            src = self.multi or r
            title = 'Overlay nhiều màu' if self.multi else f"Overlay + số vùng ({r['stats']['components']} vùng)"
            for p, img in zip(self.panels, [self.image, src['mask'], src['segmented'], src['overlay']]):
                p.show(img)
            self.panels[3].title.configure(text=title)
        else:
            hsv = r['hsv']; s = r['stats']
            imgs = [self.image, r['filtered'], hue_view(hsv), hsv[..., 1], hsv[..., 2], r['raw_mask'],
                    r['opened'], r['closed'], r['labels_view'], r['segmented'], r['overlay']]
            titles = list(PIPE_TITLES)
            titles[5] = f"B3. Mask thô ({s['raw_pixels']:,} px)"
            titles[6] = f"B4. Opening ({s['opened_pixels']:,} px)"
            titles[7] = f"B5. Closing ({s['closed_pixels']:,} px)"
            titles[8] = f"B6. Lọc vùng: {s['components_before']} → {s['components']}"
            for p, img, t in zip(self.pipe_panels, imgs, titles):
                p.show(img, t)
            last = self.pipe_panels[11]
            if self.gt is not None and self.gt.shape == r['mask'].shape:
                last.show(compare_image(r['mask'], self.gt), 'So với mask chuẩn (xanh=đúng, đỏ=thừa, lam=sót)')
            else:
                last.title.configure(text='So với mask chuẩn')
                last.clear('Chưa có mask chuẩn')

    # ---------------------------------------------------------- mở ảnh / mẫu
    def load_image(self, path, keep_gt=False):
        image = read_image(path)
        self.stop_video()
        self.image = image; self.source = str(path)
        if not keep_gt:
            self.gt = None
        self.process()

    def open_image(self):
        p = filedialog.askopenfilename(title='Mở ảnh', initialdir=str(ROOT / 'data' / 'images'),
                                       filetypes=[('Ảnh', ' '.join('*' + e for e in IMAGE_TYPES)), ('Mọi file', '*.*')])
        if p:
            self.guard(lambda: (self.load_image(p), self.sample_var.set(''), self.sample_note.configure(text='')))

    def load_sample(self, label=None):
        label = label or self.sample_var.get()
        s = self.sample_map.get(label)
        if not s:
            return

        def act():
            self.gt = read_image(s['mask'], True) if s['mask'] else None
            self.preset.set(s['target'] if s['target'] in PRESETS else 'Tùy chỉnh')
            self.changing = True
            for k, v in s['config'].to_dict().items():
                self.vars[k].set(v)
            self.changing = False
            self.load_image(s['image'], keep_gt=True)
            self.sample_var.set(label)
            self.sample_note.configure(text=s['note'] + (' • có mask chuẩn' if s['mask'] else ''))
        self.guard(act)

    def select_sample_id(self, sid):
        for label, s in self.sample_map.items():
            if s['id'] == sid:
                self.load_sample(label); return True
        return False

    # ---------------------------------------------------------- video
    def stop_video(self):
        self.playing = False; self.play_text.set('▶ Phát')
        if self.video_pending:
            self.root.after_cancel(self.video_pending); self.video_pending = None
        if self.cap is not None:
            self.cap.release()
        self.cap = None; self.frame_index = None

    def open_video(self, path=None):
        p = path or filedialog.askopenfilename(title='Mở video', initialdir=str(ROOT / 'data' / 'videos'),
                                               filetypes=[('Video', '*.avi *.mp4 *.mov *.mkv *.wmv'), ('Mọi file', '*.*')])
        if not p:
            return
        cap = cv2.VideoCapture(str(p))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok or frame is None:
            cap.release()
            messagebox.showerror('Mở video', 'Không mở được video hoặc video không có khung hình hợp lệ.'); return
        self.stop_video()
        self.cap = cap; self.source = str(p); self.image = frame; self.gt = None; self.frame_index = 0
        self.sample_var.set(''); self.sample_note.configure(text='Video: bấm ▶ Phát để chạy, tham số áp dụng ngay khi chỉnh.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.delay = max(15, round(1000 / fps)) if math.isfinite(fps) and fps > 0 else 40
        self.process()

    def toggle_play(self):
        if self.cap is None:
            messagebox.showinfo('Video', 'Hãy bấm "Mở video" trước.'); return
        self.playing = not self.playing
        self.play_text.set('⏸ Dừng' if self.playing else '▶ Phát')
        if self.playing:
            self.tick()
        elif self.video_pending:
            self.root.after_cancel(self.video_pending); self.video_pending = None

    def tick(self):
        self.video_pending = None
        if not self.playing or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:  # hết video thì quay lại đầu để demo liên tục
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); self.frame_index = 0
            ok, frame = self.cap.read()
            if not ok:
                self.stop_video(); return
        self.frame_index += 1; self.image = frame
        self.process()
        self.video_pending = self.root.after(self.delay, self.tick)

    def export(self):
        if self.export_thread and self.export_thread.is_alive():
            self.cancel_export.set(); self.status.set('Đang dừng xuất video…'); return
        if self.cap is None:
            messagebox.showinfo('Xuất video', 'Hãy mở video trước. Video được xuất với bộ tham số hiện tại.'); return
        try:
            c = self.config()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Tham số', str(exc)); return
        p = filedialog.askdirectory(title='Chọn thư mục lưu video', initialdir=str(self.output_dir()))
        if p:
            self.start_export(p, c)

    def start_export(self, folder, c):
        if self.playing:
            self.toggle_play()
        self.cancel_export.clear(); source = self.source; color = self.display_color()
        self.status.set('Đang xuất video (gốc, mask, overlay)… bấm Xuất video lần nữa để dừng.')
        self.progress['value'] = 0

        def worker():
            try:
                out, rep = export_video(source, folder, c, lambda n, t: self.events.put(('progress', n, t)),
                                        self.cancel_export, color)
                self.events.put(('done', out, rep))
            except Exception as exc:  # báo lỗi ra GUI
                self.events.put(('error', str(exc)))
        self.export_thread = threading.Thread(target=worker, daemon=True); self.export_thread.start()

    def poll_events(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == 'progress':
                    _, n, total = ev
                    self.progress['value'] = 100 * n / total if total else 0
                    self.status.set(f'Đang xuất video: {n}/{total or "?"} frame')
                elif ev[0] == 'done':
                    _, out, rep = ev
                    self.progress['value'] = 100
                    self.status.set(f"Xuất video {rep['trang_thai']}: {rep['so_frame']} frame → {out}")
                    self.last_export = out
                    if not self.closing and not getattr(self, 'silent', False):
                        messagebox.showinfo('Xuất video', self.status.get())
                else:
                    self.status.set('Lỗi xuất video: ' + ev[1])
                    if not getattr(self, 'silent', False):
                        messagebox.showerror('Xuất video', ev[1])
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(100, self.poll_events)

    # ---------------------------------------------------------- lấy màu, mask chuẩn
    def pick_color(self, x, y):
        if self.image is None:
            return
        b, g, r = map(int, self.image[y, x])
        h, s, v = map(int, cv2.cvtColor(self.image[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0])
        self.suggestion = suggest_thresholds(h, s, v)
        sg = self.suggestion
        warn = '\n⚠ S thấp: màu gần xám/trắng, H không đáng tin.' if s < 40 else ''
        self.pick_text.set(f'Pixel ({x}, {y}):  H = {h}   S = {s}   V = {v}\n'
                           f"Gợi ý: H {sg['h_min']}–{sg['h_max']}, S ≥ {sg['s_min']}, V ≥ {sg['v_min']}{warn}")
        self.swatch.configure(bg=f'#{r:02x}{g:02x}{b:02x}')

    def use_suggestion(self):
        if not self.suggestion:
            messagebox.showinfo('Lấy màu', 'Hãy nhấp vào một điểm trên ảnh gốc trước.'); return
        try:
            d = self.config().to_dict()
        except (ValueError, tk.TclError):
            d = HSVConfig().to_dict()
        d.update(self.suggestion)
        self.preset.set('Tùy chỉnh')
        self.set_config(HSVConfig(**d))

    def load_gt(self):
        if self.image is None:
            messagebox.showinfo('Mask chuẩn', 'Hãy mở ảnh trước.'); return
        if self.cap is not None:
            messagebox.showinfo('Mask chuẩn', 'Mask chuẩn chỉ dùng cho ảnh tĩnh.'); return
        p = filedialog.askopenfilename(title='Chọn mask chuẩn (ảnh trắng đen, cùng kích thước)',
                                       initialdir=str(ROOT / 'data'),
                                       filetypes=[('Ảnh', ' '.join('*' + e for e in IMAGE_TYPES)), ('Mọi file', '*.*')])
        if p:
            self.guard(lambda: self.set_gt(read_image(p, True)))

    def set_gt(self, gt):
        if gt.shape != self.image.shape[:2]:
            raise ValueError(f'Mask chuẩn {gt.shape[1]}×{gt.shape[0]} khác kích thước ảnh '
                             f'{self.image.shape[1]}×{self.image.shape[0]}.')
        self.gt = gt; self.process()

    # ---------------------------------------------------------- lưu
    def output_dir(self):
        d = ROOT / 'outputs'
        try:
            d.mkdir(exist_ok=True)
        except OSError:
            d = Path.home()
        return d

    def save(self, folder=None):
        if self.image is None or self.result is None:
            messagebox.showinfo('Lưu kết quả', 'Chưa có kết quả để lưu.'); return
        self.process()
        p = folder or filedialog.askdirectory(title='Chọn thư mục lưu kết quả', initialdir=str(self.output_dir()))
        if not p:
            return

        def act():
            out = save_result(p, self.image, self.result, self.result_config, self.source,
                              self.gt if self.gt is not None and self.gt.shape == self.result['mask'].shape else None,
                              self.frame_index)
            self.last_saved = out
            self.status.set(f'Đã lưu: {out}')
            if not getattr(self, 'silent', False):
                messagebox.showinfo('Lưu kết quả', f'Đã lưu ảnh từng bước, config.json, stats.json, components.csv vào:\n{out}')
        self.guard(act)

    def save_settings(self):
        p = filedialog.asksaveasfilename(title='Lưu cấu hình', defaultextension='.json',
                                         initialdir=str(self.output_dir()), filetypes=[('JSON', '*.json')])
        if p:
            self.guard(lambda: save_config(p, self.config(), dict(mau_muc_tieu=self.preset.get())))

    def load_settings(self):
        p = filedialog.askopenfilename(title='Nạp cấu hình', initialdir=str(ROOT / 'config'), filetypes=[('JSON', '*.json')])
        if p:
            self.guard(lambda: self.set_config(load_config(p)))

    # ---------------------------------------------------------- tiện ích
    def guard(self, action):
        try:
            action()
        except Exception as exc:
            messagebox.showerror('Không thực hiện được', str(exc))

    def report_error(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        try:
            messagebox.showerror('Lỗi', f'{val}')
        except tk.TclError:
            pass

    def help(self):
        messagebox.showinfo('Hướng dẫn', (
            '1. Chọn "Ảnh mẫu" hoặc bấm Mở ảnh / Mở video.\n'
            '2. Chọn màu cần tách (preset) rồi kéo các thanh H, S, V.\n'
            '   H min > H max nghĩa là dải vòng qua 0 (dùng cho màu đỏ).\n'
            '3. Median, Opening, Closing, diện tích tối thiểu để làm sạch mask.\n'
            '4. Nhấp ảnh gốc để xem HSV của pixel, bấm "Dùng ngưỡng gợi ý".\n'
            '5. Tab "Pipeline từng bước" hiển thị kết quả B1 → B7.\n'
            '6. Nạp mask chuẩn để tính IoU, F1 (ảnh tổng hợp có sẵn).\n'
            '7. Lưu kết quả: ảnh từng bước + config.json + stats.json + components.csv.\n'
            'Phím tắt: Ctrl+O mở ảnh, Ctrl+S lưu kết quả.'))

    def close(self):
        self.closing = True; self.cancel_export.set(); self.stop_video()
        self._wait_close()

    def _wait_close(self):
        if self.export_thread and self.export_thread.is_alive():
            self.root.after(100, self._wait_close)
        else:
            self.root.destroy()


# ------------------------------------------------------------------ tự kiểm tra
def self_test(app, out_dir):
    """Tự thao tác toàn bộ chức năng và chụp màn hình (dùng để kiểm thử, kể cả bản .exe)."""
    from PIL import ImageGrab
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    app.silent = True
    log = []
    steps = []

    def shot(name):
        app.root.update()
        x, y = app.root.winfo_rootx(), app.root.winfo_rooty()
        w, h = app.root.winfo_width(), app.root.winfo_height()
        ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True).save(out / f'{name}.png')
        log.append(f'chụp {name}.png')

    def check(cond, msg):
        log.append(('OK   ' if cond else 'LỖI  ') + msg)

    def s_all_samples():
        for s in app.samples:
            app.select_sample_id(s['id']); app.process()
            check(app.result is not None and app.source.endswith(s['image'].name), f"nạp mẫu {s['id']}: "
                  f"{app.result['stats']['foreground_percent']:.2f}% , {app.result['stats']['components']} vùng")

    def s_main():
        app.select_sample_id('01_smarties'); app.process(); app.tabs.select(0)
    def s_shot_main(): shot('01_truoc_sau_smarties')
    def s_slider():
        app.vars['s_min'].set(20); app.process()
        check(True, f"S min = 20 → {app.result['stats']['foreground_percent']:.2f}%")
    def s_shot_slider(): shot('02_s_min_thap')
    def s_reset():
        app.select_sample_id('01_smarties'); app.process()
    def s_pipeline():
        app.tabs.select(1); app.process()
    def s_shot_pipe(): shot('03_pipeline_tung_buoc')
    def s_red():
        app.tabs.select(0); app.preset.set('Đỏ'); app.apply_preset(); app.process()
        check(app.result_config.hue_wraps, f"preset Đỏ dải vòng: {app.result['stats']['components']} vùng")
    def s_shot_red(): shot('04_mau_do_hue_vong')
    def s_pick():
        app.pick_color(80, 250); app.use_suggestion(); app.process()
        check(app.preset.get() == 'Tùy chỉnh', 'lấy màu + dùng gợi ý: ' + app.pick_text.get().split('\n')[0])
    def s_multi():
        app.select_sample_id('01_smarties')
        for n in ('Đỏ', 'Xanh lá', 'Xanh dương'):
            app.multi_vars[n].set(True)
        app.process(); check(app.multi is not None, 'nhiều màu: ' + app.stats_text.get())
    def s_shot_multi(): shot('05_nhieu_mau')
    def s_multi_off():
        for v in app.multi_vars.values():
            v.set(False)
        app.process()
    def s_gt():
        app.select_sample_id('syn_01_do_hue_vong'); app.process()
        check(app.gt is not None and 'IoU' in app.metric_text.get(), app.metric_text.get())
        app.tabs.select(1); app.process()
    def s_shot_gt(): shot('06_mask_chuan_iou')
    def s_bad_gt():
        try:
            app.set_gt(np.zeros((10, 10), np.uint8)); check(False, 'mask sai kích thước không báo lỗi')
        except ValueError:
            check(True, 'mask sai kích thước bị từ chối đúng')
        try:
            read_image(Path(__file__)); check(False, 'đọc file lỗi không báo')
        except ValueError:
            check(True, 'file không phải ảnh bị từ chối đúng')
    def s_save():
        app.tabs.select(0); app.select_sample_id('01_smarties'); app.process()
        app.save(out / 'luu_thu')
        files = sorted(p.name for p in app.last_saved.iterdir())
        check('stats.json' in files and 'components.csv' in files and len(files) >= 13, f'lưu kết quả: {len(files)} file')
    def s_video():
        v = ROOT / 'data' / 'videos'
        vids = sorted(v.glob('*.avi')) + sorted(v.glob('*.mp4'))
        app.open_video(vids[0]); app.preset.set('Xanh dương'); app.apply_preset()
        app.toggle_play()
    def s_video_check():
        check(app.frame_index and app.frame_index > 3, f'phát video: frame {app.frame_index}')
    def s_shot_video(): shot('07_video')
    def s_export():
        app.toggle_play(); app.start_export(out / 'video_thu', app.config())
    def s_export_wait():
        if app.export_thread and app.export_thread.is_alive():
            return 'wait'
        app.poll_events()
        check(hasattr(app, 'last_export') and (app.last_export / 'overlay.avi').exists(), 'xuất video AVI')
    def s_done():
        (out / 'nhat_ky.txt').write_text('\n'.join(log), encoding='utf-8')
        print('\n'.join(log))
        app.close()

    steps = [s_all_samples, s_main, s_shot_main, s_slider, s_shot_slider, s_reset, s_pipeline, s_shot_pipe,
             s_red, s_shot_red, s_pick, s_multi, s_shot_multi, s_multi_off, s_gt, s_shot_gt, s_bad_gt, s_save,
             s_video, None, s_video_check, s_shot_video, s_export, s_export_wait, s_done]

    def run(i=0):
        if i >= len(steps):
            return
        st = steps[i]
        try:
            res = st() if st else None
        except Exception as exc:
            log.append(f'LỖI  bước {getattr(st, "__name__", i)}: {exc!r}')
            res = None
        if res == 'wait':
            app.root.after(300, lambda: run(i)); return
        app.root.after(1500 if st is None else 500, lambda: run(i + 1))
    app.root.after(1200, run)


def main(argv=None):
    ap = argparse.ArgumentParser(description='HSV Color Segmentation')
    ap.add_argument('--image', type=Path, help='mở ngay một ảnh')
    ap.add_argument('--video', type=Path, help='mở ngay một video')
    ap.add_argument('--sample', default='01_smarties', help='ID ảnh mẫu (tên file trong config/)')
    ap.add_argument('--tu-kiem-tra', dest='self_test', type=Path, help='tự thao tác + chụp màn hình vào thư mục này')
    args = ap.parse_args(argv)
    try:  # chữ sắc nét trên màn hình có scale 125–150%
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    app = HSVApplication(root)

    def initial():
        if args.video:
            app.open_video(args.video)
        elif args.image:
            app.guard(lambda: app.load_image(args.image))
        elif not app.select_sample_id(args.sample):
            app.status.set('Không tìm thấy ảnh mẫu: ' + args.sample)
    root.after(200, initial)
    if args.self_test:
        self_test(app, args.self_test)
    root.mainloop()
