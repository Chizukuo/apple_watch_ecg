# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Chizukuo
"""
合成测试数据：不含任何真实个人信息。

- make_beats / synth_signal：用高斯波叠加生成 30 秒单导联心电，可插入室性早搏（负向/正向）、
  房性早搏、窦性停搏、RR 长短交替。
- build_apple_like_pdf：按 Apple Watch 导出 PDF 的版式画出矢量 PDF（三条网格带、定标方波、
  被切成多段的红色波形、页眉页脚文字、文件属性），用来端到端测试提取流程。
"""
from __future__ import annotations

import numpy as np
import pymupdf

FS = 512
PT_PER_MM = 72.0 / 25.4


def _g(t, a, mu, sd):
    return a * np.exp(-0.5 * ((t - mu) / sd) ** 2)


# (名称, 振幅 mV, 相对 R 峰的时间 s, 宽度 s)
WAVES = {
    "N": [("P", 0.10, -0.16, 0.022), ("Q", -0.06, -0.028, 0.008), ("R", 0.95, 0.0, 0.009),
          ("S", -0.30, 0.024, 0.009), ("T", 0.30, 0.25, 0.045)],
    "A": [("P", -0.06, -0.13, 0.020), ("Q", -0.06, -0.028, 0.008), ("R", 0.95, 0.0, 0.009),
          ("S", -0.30, 0.024, 0.009), ("T", 0.30, 0.25, 0.045)],                 # 房性早搏：P 波不同，QRS 相同
    "V": [("R", 0.35, -0.04, 0.025), ("S", -1.40, 0.02, 0.032), ("T", 0.45, 0.30, 0.06)],   # 室性早搏（负向）
    "W": [("R", 1.30, 0.0, 0.032), ("S", -0.45, 0.06, 0.025), ("T", -0.45, 0.30, 0.06)],    # 室性早搏（正向）
}


def waves_variant(t_shift=0.0, qrs_scale=1.0, p_at=-0.16, st=0.0, t_amp=0.30, t_sd=0.045):
    """在正常心搏基础上改形态：T 波后移（QT 变长）、QRS 加宽、P 波位置（PR）、ST 段抬高/压低、T 波幅度/方向。"""
    k = qrs_scale
    n = [("P", 0.10, p_at, 0.022), ("Q", -0.06, -0.028 * k, 0.008 * k), ("R", 0.95, 0.0, 0.009 * k),
         ("S", -0.30, 0.024 * k, 0.009 * k), ("T", t_amp, 0.25 + t_shift, t_sd)]
    if st:
        n.append(("ST", st, 0.03 * k + 0.06, 0.06))
    return {**WAVES, "N": n}


def make_beats(duration=30.0, hr=72.0, events=None, alt=None, start=0.36, first=0.5):
    """
    返回 [(时间, 类型)]。
    events: {窦性心搏序号: "V" | "W" | "A" | "pause"}，在该窦性心搏之后插入事件。
    alt: (起, 止) 窦性心搏序号区间内 RR 长短交替（0.88 / 1.12 倍）。
    """
    events = events or {}
    rr = 60.0 / hr
    beats, tcur, k = [], start + first, 0
    while tcur < duration - 0.3:
        ev = events.get(k)
        beats.append((tcur, "N"))
        if ev in ("V", "W"):
            beats.append((tcur + 0.6 * rr, ev))
            tcur += 2 * rr                           # 代偿间歇
        elif ev == "A":
            beats.append((tcur + 0.7 * rr, "A"))
            tcur += 0.7 * rr + rr                    # 窦房结重整，非代偿
        elif ev == "pause":
            tcur += 2.2 * rr                         # 窦性停搏
        elif alt and alt[0] <= k < alt[1]:
            tcur += rr * (0.88 if k % 2 == 0 else 1.12)
        else:
            tcur += rr
        k += 1
    return [(t, kind) for t, kind in beats if t < duration - 0.3]


def make_rsa_beats(hr=70.0, depth=0.15, period=4.0, duration=30.0, start=0.36, seed=0):
    """呼吸性窦性心律不齐：RR 随呼吸（period 秒）在 ±depth 范围内渐变，另加 10 ms 随机抖动。"""
    rng = np.random.default_rng(seed)
    rr0 = 60.0 / hr
    t, out = start + 0.3 + rng.uniform(0, rr0), []
    while t < duration - 0.3:
        out.append((t, "N"))
        t += rr0 * (1 + depth * np.sin(2 * np.pi * t / period)) + rng.normal(0, 0.01)
    return out


def add_motion_artifact(t, v, at=14.0, amp=1.2, width=0.25):
    """叠加一个缓慢的大偏移（手指移动、压力变化造成的基线漂移）。"""
    return v + amp * np.exp(-0.5 * ((t - at) / width) ** 2)


def synth_signal(beats, duration=30.0, fs=FS, noise=0.008, wander=0.03, seed=0, waves=None):
    waves = waves or WAVES
    t = np.arange(0, duration, 1.0 / fs)
    v = np.zeros_like(t)
    for t0, kind in beats:
        m = (t > t0 - 0.45) & (t < t0 + 0.8)
        v[m] += sum(_g(t[m], a, t0 + mu, sd) for _, a, mu, sd in waves[kind])
    rng = np.random.default_rng(seed)
    v += noise * rng.standard_normal(t.size) + wander * np.sin(2 * np.pi * 0.25 * t)
    return t, v


def build_apple_like_pdf(path, t, v, *, name="测试用户", dob="2000年1月1日", age=26,
                         recorded="2026年10月13日11:25", label="窦性心律", bpm=72,
                         summary="此心电图未显示房颤迹象。", algorithm=2, symptoms=None,
                         start=0.36, fs=FS, creation="D:20261013033049Z00'00'"):
    doc = pymupdf.open()
    page = doc.new_page(width=792, height=612)
    font = "china-s"
    page.insert_text((40, 50), name, fontname=font, fontsize=14)
    page.insert_text((40, 66), f"出生日期：{dob}（{age}岁）", fontname=font, fontsize=9)
    page.insert_text((560, 66), f"记录时间：{recorded}", fontname=font, fontsize=9)
    page.insert_text((40, 95), f"{label} —   平均{bpm}次/分 ", fontname=font, fontsize=10)
    page.insert_text((40, 112), summary, fontname=font, fontsize=9)
    if symptoms:
        page.insert_text((40, 128), f"症状：{symptoms}", fontname=font, fontsize=9)
    page.insert_text((40, 560), f"25 mm/s, 10 mm/mV, 导联I, {fs}Hz, iOS 27.0.1, watchOS 27.0.1, Watch8,5, "
                                f"算法版本 {algorithm} — 波形与导联I心电图相似。", fontname=font, fontsize=6)

    pt_s, pt_mv = 25 * PT_PER_MM, 10 * PT_PER_MM
    x0 = 40.0
    x1 = x0 + 10 * pt_s
    strips = [(218.5, 304.0), (323.0, 408.5), (427.6, 513.1)]
    sh = page.new_shape()
    for y0, y1 in strips:
        for major in (False, True):                                   # 细格 0.9 灰、粗格 0.8 灰
            for i in range(0, 251):
                if (i % 5 == 0) == major:
                    x = x0 + i * PT_PER_MM
                    sh.draw_line((x, y0), (x, y1))
            for j in range(0, int((y1 - y0) / PT_PER_MM) + 1):
                if (j % 5 == 0) == major:
                    y = y0 + j * PT_PER_MM
                    sh.draw_line((x0, y), (x1, y))
            g = 0.8 if major else 0.9
            sh.finish(color=(g, g, g), width=0.25)
    # 定标方波（1 mV）
    y0, y1 = strips[0]
    base = (y0 + y1) / 2
    sh.draw_polyline([(x0, base), (x0 + 4.25, base), (x0 + 4.25, base - pt_mv), (x0 + 21.25, base - pt_mv),
                      (x0 + 21.25, base), (x0 + 25.5, base)])
    sh.finish(color=(0.4, 0.4, 0.4), width=1, closePath=False)
    # 红色波形：每行切成多段路径（与 Apple 的导出方式一致）
    for k, (y0, y1) in enumerate(strips):
        m = (t >= max(start, 10 * k)) & (t < 10 * (k + 1))
        if not m.any():
            continue
        xs = x0 + (t[m] - 10 * k) * pt_s
        ys = np.clip((y0 + y1) / 2 - v[m] * pt_mv, y0, y1)
        pts = list(zip(xs.tolist(), ys.tolist()))
        for a in range(0, len(pts) - 1, 127):
            sh.draw_polyline(pts[a:a + 128])
            sh.finish(color=(0.804, 0.039, 0.125), width=1, closePath=False)
    sh.commit()
    doc.set_metadata({"producer": "iOS Version 27.0.1 (Build 24A446) Quartz PDFContext",
                      "creationDate": creation, "modDate": creation})
    doc.save(str(path))
    return path
