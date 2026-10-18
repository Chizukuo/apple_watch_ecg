#!/usr/bin/env python3
"""
apple_watch_ecg.py —— Apple Watch 心电图 PDF / 截图 → 连续信号 → 节律、形态、HRV 报告

用法:
    python apple_watch_ecg.py 记录.pdf [-o 输出目录]
    python apple_watch_ecg.py 文件夹/ [-o 输出目录]            # 批量，另外输出 summary.csv
    python apple_watch_ecg.py 记录.pdf --exclude 0-4,27.5-30   # 手动排除时段（只能用于单个文件）
    python apple_watch_ecg.py 记录.pdf --include-identity      # 在报告中保留姓名、出生日期、源文件名

隐私:
    默认输出不含姓名、出生日期和源文件名（Apple 导出的文件名里带姓名）。
    输出按 PDF 内的记录时间命名，例如 ECG_20261013_1125；没有文字层的图片按文件内容哈希命名。

处理流程:
    1. 读取 PDF 文字层：记录时间、出生日期（→ 记录时年龄）、手表分类与平均心率、算法版本、
       纸速/增益/采样率、设备与系统版本、补录的症状；文件属性：导出时间、iOS 构建号。
    2. 矢量 PDF：按灰色网格条带把红色路径归到三行，按纸速换算时间、按定标方波（1 mV）换算电压；
       没有矢量时退回光栅识别（截图、扫描件）。
    3. 三行在时间上是连续的 30 秒，拼接成一条信号后再检测心搏（避免行首尾漏检或误检）。
    4. 心搏检测 = 常规检测 + 长间隔中段补检 + 远离已知心搏的大偏折检测（宽大畸形的搏动常被常规检测漏掉）。
    5. 逐拍标注：正常 / 提前·QRS同形 / 提前·QRS异形 / 形态异常 / 排除段内；识别长间歇、代偿间歇、
       二联律样、RR 长短交替。被排除的时段不会被当成停顿。
    6. HRV 用全部 NN 间期（两端都是正常心搏）；形态筛选只决定哪些心搏进入平均模板。
    7. 在平均心搏上测量 P、PR、QRS、ST、T、QT/QTc；测不准的量明确标注。
    8. 与手表结果对照：心率、分类（按算法版本的心率范围）、年龄是否在 Apple 说明的适用范围内。

仅供个人记录与对比，不是医疗诊断。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import cv2
import neurokit2 as nk
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import detrend, savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei",
                                   "SimHei", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", module="neurokit2")

__version__ = "1.0.0"

DEFAULT_FS = 512
PT_PER_MM = 72.0 / 25.4
STRIP_SECONDS = 10.0

# ---- 节律判定阈值 ------------------------------------------------------------
PREMATURE_RATIO = 0.80      # 前一个 RR < 局部参考 RR × 此值，视为提前
ABRUPT_RATIO = 0.88         # 且比再前一个 RR 突然缩短到此比例以下（呼吸引起的变化是渐变的，不算）
PAUSE_RATIO = 1.70          # RR ≥ 局部参考 RR × 此值，视为长间歇
SAME_QRS_CORR = 0.85        # 提前搏动的 QRS 与主导形态相关系数 ≥ 此值，视为"同形"
ODD_QRS_CORR = 0.70         # 不提前、但 QRS 相关系数低于此值，视为"形态异常"
WIDE_QRS_DELTA_MS = 40      # 单拍 QRS 比主导形态宽出此值以上，视为"异形"
TEMPLATE_CORR = 0.90        # 进入平均模板的最低相关系数（QRS–T 段）
ALT_MIN_DELTA_MS = 80       # 长短交替：相邻 RR 的最小变化
T_GUARD_MAX_S = 0.42        # 误检 T 波：距上一拍不足此值
T_GUARD_SLOPE = 0.33        # 且斜率不到上一拍的此比例
ARTIFACT_MV = 0.5           # 1 秒内低频成分峰峰值超过此值，视为伪影

LABEL_NORMAL = "正常"
LABEL_PREM_SAME = "提前·QRS同形"
LABEL_PREM_DIFF = "提前·QRS异形"
LABEL_ODD = "形态异常"
LABEL_REMOVED = "排除段内"

# Apple ECG App 2.0 说明书：各分类对应的心率范围（次/分）
WATCH_RANGES_V2 = {"sinus": (50, 99), "high_hr": (100, 150), "afib": (50, 99), "afib_high": (100, 150)}
APPLE_MIN_AGE = 22          # Apple 说明书：心电图 App 不适用于 22 岁以下人群

# 形态提醒阈值（依据见 README“形态提醒”一节）
QTC_LONG = {"male": 450, "female": 460, "unknown": 450}   # AHA/ACCF/HRS 2009；性别未知时取较低（更敏感）的男性值
QTC_LONG_MARKED = 500       # 明显延长，与心律失常风险显著相关
QTC_SHORT = 360             # ESC 2015：≤360 ms 且伴其他依据时可诊断短 QT 综合征
QTC_SHORT_MARKED = 340      # ESC 2015：≤340 ms 即可诊断
QTC_FRIDERICIA_ABOVE_HR = 100   # 心率 > 100 时 Bazett 过度校正，改用 Fridericia
QT_MIN_T_MV = 0.08          # T 波低于此幅度时 T 终点不可靠，不做 QT 提醒
QT_MAX_IQR_MS = 40          # 逐拍 QT 四分位距超过此值：测量不稳定，不做 QT 提醒
QRS_WIDE_MS = 120
PR_LONG_MS = 200
PR_SHORT_MS = 120
ST_DEV_MV = 0.10            # 抬高看 J 点（第四版心肌梗死通用定义的测量点）；压低要求 J 点和 J+60 ms 都低于此值
T_INV_MIN_MV = 0.10
MIN_TEMPLATE_BEATS = 5
PR_MAX_HR = 100             # 心率更快时 P 波常落在上一拍的 T 波上，不做 PR 提醒
SEXES = ("male", "female", "unknown")
PLACEMENTS = ("wrist", "other")
SKIP_QT = "未做 QTc 提醒"
ALERT_NAMES = {"QTC_LONG_MARKED": "QTc明显延长", "QTC_LONG": "QTc偏长", "QTC_SHORT_MARKED": "QTc明显偏短",
               "QTC_SHORT": "QTc偏短", "QRS_WIDE": "QRS增宽", "PR_LONG": "PR延长", "PR_SHORT": "PR偏短",
               "ST_ELEVATION": "ST抬高", "ST_DEPRESSION": "ST压低", "T_INVERSION": "T波倒置"}


class AnalysisError(RuntimeError):
    """记录无法分析（心搏太少、找不到波形等）。"""


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------
@dataclass
class Row:
    index: int                 # 第几行（0, 1, 2）
    t: np.ndarray              # 绝对时间（秒，从第一行网格左端算起）
    v: np.ndarray              # 电压（mV，已减去行中位数）
    clip: np.ndarray           # 逐点：是否碰到网格上下边缘（被截断）


@dataclass
class Recording:
    path: Path
    watch: dict                # 来自 PDF 的手表信息（不含身份信息）
    identity: dict             # 姓名、出生日期、源文件名（默认不输出）
    rows: list[Row]
    source: str                # "vector" | "raster"
    fs: int
    record_id: str
    t: np.ndarray              # 拼接后的连续信号
    v: np.ndarray
    clip: np.ndarray
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# PDF 文字层与文件属性
# ----------------------------------------------------------------------------
def watch_category(label: str | None) -> str:
    """把手表分类文字归到固定类别；认不出的返回 unknown。"""
    if not label:
        return "unknown"
    if "房颤" in label or "心房颤动" in label:
        return "afib_high" if "高" in label else "afib"
    if "窦性" in label:
        return "sinus"
    if ("低" in label and "高" in label) or "过低" in label:
        return "low_or_high"
    if "高心率" in label or "心率较高" in label or "心率过高" in label:
        return "high_hr"
    if "不确定" in label or "无法" in label:
        return "inconclusive"
    if "不佳" in label or "较差" in label:
        return "poor"
    return "unknown"


def parse_pdf_text(text: str) -> tuple[dict, dict]:
    """返回 (watch, identity)。watch 可以写进报告；identity 默认不写。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    watch = {
        "recorded_at": None, "age_at_recording": None, "age_shown": None,
        "classification": None, "category": "unknown", "reported_bpm": None,
        "summary_text": None, "symptoms": None,
        "paper_speed_mm_s": 25.0, "gain_mm_mv": 10.0, "sampling_rate_hz": DEFAULT_FS, "lead_label": None,
        "device_model": None, "ios_version": None, "watchos_version": None, "algorithm_version": None,
        "exported_at": None, "ios_build": None,
    }
    identity = {"name": lines[0] if lines else None, "dob": None}

    m = re.search(r"记录时间[:：]\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[:：](\d{2})", text)
    rec_dt = datetime(*map(int, m.groups())) if m else None
    if rec_dt:
        watch["recorded_at"] = rec_dt.strftime("%Y-%m-%d %H:%M")
    m = re.search(r"出生日期[:：]\s*(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    dob = date(*map(int, m.groups())) if m else None
    if dob:
        identity["dob"] = dob.isoformat()
        if rec_dt:
            d = rec_dt.date()
            watch["age_at_recording"] = d.year - dob.year - ((d.month, d.day) < (dob.month, dob.day))

    pats = {
        "age_shown": (r"[（(]\s*(\d+)\s*岁\s*[)）]", int),
        "reported_bpm": (r"平均\s*(\d+)\s*次\s*/\s*分", int),
        "paper_speed_mm_s": (r"(\d+(?:\.\d+)?)\s*mm/s", float),
        "gain_mm_mv": (r"(\d+(?:\.\d+)?)\s*mm/mV", float),
        "sampling_rate_hz": (r"(\d+)\s*Hz", int),
        "lead_label": (r"(导联\s*[IVX]+)", str),
        "ios_version": (r"iOS\s*([\d.]+\d)", str),
        "watchos_version": (r"watchOS\s*([\d.]+\d)", str),
        "device_model": (r"(Watch\d+,\d+)", str),
        "algorithm_version": (r"算法版本\s*(\d+)", int),
    }
    for key, (pat, conv) in pats.items():
        m = re.search(pat, text)
        if m:
            watch[key] = conv(m.group(1).strip())

    # 分类：标题行"窦性心律 — 平均86次/分"中"—"之前的部分
    for ln in lines:
        if "次/分" in ln and "—" in ln:
            watch["classification"] = ln.split("—")[0].strip()
            break
    watch["category"] = watch_category(watch["classification"])
    # 手表的文字结论可能跨多行（例如"高心率"会附带就医建议），一直取到参数行之前
    start = next((k for k, ln in enumerate(lines) if ln.startswith("此心电图") or "房颤迹象" in ln), None)
    if start is not None:
        end = next((k for k in range(start, len(lines)) if "mm/s" in lines[k]), len(lines))
        watch["summary_text"] = "".join(lines[start:end])
    m = re.search(r"症状[:：]\s*(.+)", text)
    watch["symptoms"] = m.group(1).strip() if m else None
    return watch, identity


def parse_pdf_properties(doc) -> dict:
    """PDF 文件属性：导出时间（换算为本机时区）和 iOS 构建号。"""
    md = doc.metadata or {}
    out = {"exported_at": None, "ios_build": None}
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})Z", md.get("creationDate") or "")
    if m:
        t = datetime(*map(int, m.groups()), tzinfo=timezone.utc).astimezone()
        out["exported_at"] = t.strftime("%Y-%m-%d %H:%M (UTC%z)")
    m = re.search(r"Build\s+(\w+)", md.get("producer") or "")
    if m:
        out["ios_build"] = m.group(1)
    return out


def make_record_id(watch: dict, path: Path) -> str:
    if watch.get("recorded_at"):
        return "ECG_" + watch["recorded_at"].replace("-", "").replace(":", "").replace(" ", "_")
    return "ECG_" + hashlib.sha1(path.read_bytes()).hexdigest()[:8]


# ----------------------------------------------------------------------------
# 矢量 PDF 提取
# ----------------------------------------------------------------------------
def _is_red(c) -> bool:
    return bool(c) and len(c) == 3 and c[0] > 0.6 and c[1] < 0.35 and c[2] < 0.35


def _is_gray(c, lo: float, hi: float) -> bool:
    return bool(c) and len(c) == 3 and max(c) - min(c) < 0.03 and lo <= c[0] <= hi


def _line_points(d):
    """路径中的点：直线取两端，贝塞尔曲线取起止点。"""
    for it in d["items"]:
        if it[0] == "l":
            yield it[1]; yield it[2]
        elif it[0] == "c":
            yield it[1]; yield it[4]


def _grid_strips(drawings) -> list[tuple[float, float, float, float]]:
    """由浅灰网格线得到每条波形带的 (x0, y0, x1, y1)，按 y 排序。"""
    segs = []
    for d in drawings:
        if not _is_gray(d.get("color"), 0.75, 0.95):
            continue
        for it in d["items"]:
            if it[0] == "l":
                segs.append((it[1].x, it[1].y, it[2].x, it[2].y))
    if not segs:
        return []
    segs = np.array(segs)
    ys = np.sort(np.concatenate([segs[:, 1], segs[:, 3]]))
    cuts = np.where(np.diff(ys) > 5.0)[0]                      # 条带之间有明显空白
    strips = []
    for b in np.split(ys, cuts + 1):
        y0, y1 = b.min(), b.max()
        m = (np.minimum(segs[:, 1], segs[:, 3]) >= y0 - 0.1) & (np.maximum(segs[:, 1], segs[:, 3]) <= y1 + 0.1)
        if not m.any():                                         # 例如跨条带的刻度短线
            continue
        x0 = np.minimum(segs[m, 0], segs[m, 2]).min()
        x1 = np.maximum(segs[m, 0], segs[m, 2]).max()
        if y1 - y0 > 20 and x1 - x0 > 200:
            strips.append((x0, y0, x1, y1))
    return sorted(strips, key=lambda s: s[1])


def _cal_pulse_height_pt(drawings, strip) -> float | None:
    """第一条带左端的深灰定标方波，其高度 = 1 mV。"""
    x0, y0, x1, y1 = strip
    best = None
    for d in drawings:
        c = d.get("color")
        if not (c and len(c) == 3 and max(c) - min(c) < 0.03 and c[0] < 0.6):
            continue
        r = d["rect"]
        if r.y0 >= y0 - 1 and r.y1 <= y1 + 1 and r.x0 <= x0 + 5 and r.width < 60 and 10 < r.height < 60:
            best = r.height
    return best


def extract_vector(page, watch: dict, fs: int) -> tuple[list[Row], list[str]] | None:
    drawings = page.get_drawings()
    red = [d for d in drawings if _is_red(d.get("color")) and len(d["items"]) >= 4]
    strips = _grid_strips(drawings)
    if len(red) < 10 or not strips:
        return None
    notes = []
    pt_per_s = watch["paper_speed_mm_s"] * PT_PER_MM
    cal = _cal_pulse_height_pt(drawings, strips[0])
    expect = watch["gain_mm_mv"] * PT_PER_MM
    pt_per_mv = cal if cal else expect
    if cal and abs(cal - expect) / expect > 0.03:
        notes.append(f"定标方波高度 {cal:.1f} pt 与增益 {watch['gain_mm_mv']} mm/mV 不一致，已按定标方波换算")
    if not cal:
        notes.append("未找到定标方波，电压按 PDF 标注的增益换算")
    rows = []
    for i, (x0, y0, x1, y1) in enumerate(strips):
        strip_dur = (x1 - x0) / pt_per_s
        if abs(strip_dur - STRIP_SECONDS) > 0.2:
            notes.append(f"第 {i + 1} 行网格宽度换算为 {strip_dur:.2f} 秒（预期 10 秒），请核对纸速")
        pts = []
        for d in red:
            cy = (d["rect"].y0 + d["rect"].y1) / 2
            if y0 - 2 <= cy <= y1 + 2:
                pts.extend((p.x, p.y) for p in _line_points(d))
        if len(pts) < 50:
            continue
        pts = np.array(pts)
        xs, idx = np.unique(pts[:, 0], return_index=True)        # 已按 x 排序
        ys = pts[idx, 1]
        t_abs = i * strip_dur + (xs - x0) / pt_per_s
        t = np.arange(t_abs[0], t_abs[-1], 1.0 / fs)
        y = np.interp(t, t_abs, ys)
        rows.append(Row(i, t, -(y - np.median(y)) / pt_per_mv, (y >= y1 - 0.6) | (y <= y0 + 0.6)))
    return (rows, notes) if rows else None


# ----------------------------------------------------------------------------
# 光栅图片提取（截图、扫描件、不含矢量的 PDF）
# ----------------------------------------------------------------------------
def _runs(mask_1d: np.ndarray, min_len: int):
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(g[0], g[-1]) for g in groups if g[-1] - g[0] + 1 >= min_len]


def extract_raster(img_bgr: np.ndarray, watch: dict, fs: int) -> tuple[list[Row], list[str]]:
    notes = []
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((H <= 12) | (H >= 165)) & (S > 90) & (V > 70)
    grid = (S < 35) & (V > 165) & (V < 245)
    dark = (S < 40) & (V < 160)

    # 1) 网格条带：网格像素占行宽比例高的连续行；波形横穿网格会遮住一两行，间隙很小的段合并
    bands, merged = _runs(grid.sum(axis=1) / w > 0.06, 2), []
    for b0, b1 in bands:
        if merged and b0 - merged[-1][1] <= max(4, h // 100):
            merged[-1] = (merged[-1][0], b1)
        else:
            merged.append((b0, b1))
    bands = [b for b in merged if b[1] - b[0] >= max(10, h // 40)]
    bands = sorted(sorted(bands, key=lambda b: b[1] - b[0], reverse=True)[:3])
    if not bands:
        raise AnalysisError("图片中找不到心电图网格")

    # 2) 每条带的左右边界（竖直网格线等间距，按间隙聚类取最宽的一簇，避开截图边框）
    geo, px_per_s_all, cal_px = [], [], None
    for i, (by0, by1) in enumerate(bands):
        cols = np.flatnonzero(grid[by0:by1 + 1].sum(axis=0) / (by1 - by0 + 1) > 0.5)
        if cols.size < 2:
            continue
        groups = np.split(cols, np.where(np.diff(cols) > max(6, w // 30))[0] + 1)
        g = max(groups, key=lambda c: c[-1] - c[0])
        gx0, gx1 = int(g[0]), int(g[-1])
        px_per_s = (gx1 - gx0) / STRIP_SECONDS
        px_per_s_all.append(px_per_s)
        if i == 0:                                              # 定标方波：第一条带左端的深灰竖线
            rr = np.flatnonzero(dark[by0:by1 + 1, gx0:gx0 + int(0.6 * px_per_s)].any(axis=1))
            if rr.size and 0.25 * px_per_s < rr.max() - rr.min() < 0.8 * px_per_s:
                cal_px = rr.max() - rr.min()
        geo.append((i, by0, by1, gx0, gx1))
    if not px_per_s_all:
        raise AnalysisError("图片中找不到网格的左右边界")
    px_per_s = float(np.median(px_per_s_all))
    if cal_px:
        px_per_mv = float(cal_px)
    else:
        px_per_mv = px_per_s / watch["paper_speed_mm_s"] * watch["gain_mm_mv"]
        notes.append("图片中未找到定标方波，电压按纸速和增益推算")
    notes.append(f"光栅提取：{px_per_s:.1f} 像素/秒，{px_per_mv:.1f} 像素/mV；精度低于矢量 PDF。"
                 "已知偏差：QRS 可能偏宽（最多约 25 ms），截断无法可靠判断")

    rows = []
    for i, by0, by1, gx0, gx1 in geo:
        band = red[by0:by1 + 1, gx0:gx1 + 1]
        if not band.any():
            continue
        prev = np.median(np.flatnonzero(band.any(axis=1)))
        xs, ys = [], []
        for x in range(band.shape[1]):
            col = np.flatnonzero(band[:, x])
            if col.size == 0:
                continue
            if col[-1] - col[0] <= 2:
                xs.append(x); ys.append(col.mean()); prev = col.mean()
            else:
                # 陡峭段：一列里线条从"入口"走到"出口"，两端都保留，峰顶和谷底不会被削平
                near, far = (col[0], col[-1]) if abs(col[0] - prev) <= abs(col[-1] - prev) else (col[-1], col[0])
                xs += [x - 0.25, x + 0.25]; ys += [near, far]; prev = far
        if len(xs) < 50:
            continue
        xs, ys = np.array(xs, float), np.array(ys, float)
        t_abs = i * STRIP_SECONDS + xs / px_per_s
        t = np.arange(t_abs[0], t_abs[-1], 1.0 / fs)
        y = np.interp(t, t_abs, ys)
        rows.append(Row(i, t, -(y - np.median(y)) / px_per_mv, (y <= 1) | (y >= band.shape[0] - 2)))
    if not rows:
        raise AnalysisError("图片中找不到红色波形")
    return rows, notes


# ----------------------------------------------------------------------------
# 拼接与加载
# ----------------------------------------------------------------------------
def stitch(rows: list[Row], fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三行是同一段连续记录：按时间拼接，并消除各行减去不同中位数造成的台阶。"""
    ts, vs, cs = [], [], []
    for r in sorted(rows, key=lambda r: r.t[0]):
        t, v, c = r.t, r.v.copy(), r.clip.astype(float)
        if ts:
            keep = t > ts[-1][-1]                               # 防止相邻两行在时间上重叠
            t, v, c = t[keep], v[keep], c[keep]
            if t.size == 0:
                continue
            v = v - v[0] + vs[-1][-1]
        ts.append(t); vs.append(v); cs.append(c)
    t_all, v_all, c_all = np.concatenate(ts), np.concatenate(vs), np.concatenate(cs)
    t_u = np.arange(t_all[0], t_all[-1], 1.0 / fs)
    v_u = np.interp(t_u, t_all, v_all)
    return t_u, v_u - np.median(v_u), np.interp(t_u, t_all, c_all) > 0.5


def load(path: Path) -> Recording:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        with pymupdf.open(path) as doc:                            # 及时关闭，批量时不占着文件句柄
            watch, identity = parse_pdf_text("\n".join(p.get_text("text") for p in doc))
            watch.update(parse_pdf_properties(doc))
            fs = watch["sampling_rate_hz"] if 100 <= (watch["sampling_rate_hz"] or 0) <= 2000 else DEFAULT_FS
            res = extract_vector(doc[0], watch, fs)
            if res:
                rows, notes = res
                source = "vector"
            else:
                pix = doc[0].get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
                rows, notes = extract_raster(img, watch, fs)
                source = "raster"
    else:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)   # 支持中文路径
        if img is None:
            raise AnalysisError(f"无法读取图片：{path.name}")
        watch, identity = parse_pdf_text("")
        identity["name"] = None
        fs = DEFAULT_FS
        rows, notes = extract_raster(img, watch, fs)
        notes.append("图片没有文字层：记录时间、手表结论等信息为空")
        source = "raster"
    identity["source_file"] = path.name
    t, v, clip = stitch(rows, fs)
    return Recording(path, watch, identity, rows, source, fs, make_record_id(watch, path), t, v, clip, notes)


# ----------------------------------------------------------------------------
# 心搏检测
# ----------------------------------------------------------------------------
def _odd(n: int) -> int:
    n = max(5, int(n))
    return n if n % 2 else n + 1


def _in_any(x: float, ivs) -> bool:
    return any(a <= x <= b for a, b in ivs)


def _overlaps(a0: float, a1: float, ivs) -> bool:
    return any(not (b1 < a0 or b0 > a1) for b0, b1 in ivs)


def _local_max(arr: np.ndarray, i: int, half: int) -> float:
    return float(arr[max(0, i - half):i + half + 1].max())


def _baseline(v: np.ndarray, fs: int) -> np.ndarray:
    """两级中值滤波（0.2 s 去 QRS，0.6 s 去 P/T 波）估计基线。"""
    b = median_filter(v, size=_odd(0.2 * fs), mode="nearest")
    return median_filter(b, size=_odd(0.6 * fs), mode="nearest")


def _artifact_windows(t: np.ndarray, v: np.ndarray, fs: int) -> list[tuple[float, float]]:
    """基线在任意 1 秒内的峰峰值超过阈值，视为伪影（手指移动、压力变化等）。窗口每 0.25 秒滑动一次；
    合并后再收缩到基线明显偏离（> 阈值的一半）的那一段，两侧各留 0.2 秒。"""
    base = _baseline(v, fs)
    win, step = fs, max(1, fs // 4)
    spans = []
    for s in range(0, max(1, len(base) - win + 1), step):
        if np.ptp(base[s:s + win]) > ARTIFACT_MV:
            e = min(s + win, len(base) - 1)
            if spans and s <= spans[-1][1]:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
    ref = float(np.median(base))
    out = []
    for s, e in spans:
        dev = np.flatnonzero(np.abs(base[s:e + 1] - ref) > ARTIFACT_MV / 2)
        if dev.size:
            s, e = s + dev[0], s + dev[-1]
        a = float(t[max(0, s - int(0.2 * fs))])
        b = float(t[min(len(t) - 1, e + int(0.2 * fs))])
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _clean_and_detect(v: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """常规检测。两端用常数延拓 1 秒，让滤波器的起止过渡落在延拓段里。"""
    pad = fs
    vv = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
    clean = nk.ecg_clean(vv, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    pk = np.asarray(info["ECG_R_Peaks"], int) - pad
    clean = clean[pad:pad + len(v)]
    return pk[(pk >= 0) & (pk < len(v))], clean


def _snap(i: int, clean: np.ndarray, fs: int) -> int:
    """对齐到 ±50 ms 内绝对值最大的偏折。"""
    w = int(0.05 * fs)
    a, b = max(0, i - w), min(len(clean), i + w + 1)
    return a + int(np.argmax(np.abs(clean[a:b])))


def detect_beats(t, v, fs, removed) -> tuple[list[int], dict[int, str], np.ndarray, np.ndarray]:
    raw, clean = _clean_and_detect(v, fs)
    slope = np.abs(np.gradient(clean)) * fs
    cand = {}
    for p in raw:
        cand.setdefault(_snap(int(p), clean, fs), "检测")
    if len(cand) < 3:
        raise AnalysisError("检测到的心搏少于 3 个，无法分析")
    keys = sorted(cand)
    amp_ref = float(np.median(np.abs(clean[keys])))
    slope_ref = float(np.median([_local_max(slope, p, int(0.06 * fs)) for p in keys]))

    # 补检：间隔 > 1.5 × 局部中位数时，在间隔中段用较低门槛再找一次（漏检的心搏、早搏后的代偿间歇）
    b = np.array(keys)
    rr = np.diff(b)
    for k in range(len(rr)):
        local = np.median(rr[max(0, k - 4):k + 5])
        if rr[k] <= 1.5 * local:
            continue
        a, z = b[k] + int(0.25 * fs), b[k + 1] - int(0.25 * fs)
        if z - a < int(0.05 * fs):
            continue
        i = a + int(np.argmax(np.abs(clean[a:z])))
        if _in_any(t[i], removed):
            continue
        if abs(clean[i]) >= 0.4 * amp_ref and _local_max(slope, i, int(0.06 * fs)) >= 0.25 * slope_ref:
            cand.setdefault(_snap(i, clean, fs), "补检")

    # 大偏折：远离已知心搏（前 0.2 s 到后 0.35 s 以外）、幅度明显大于普通 QRS 的偏折
    mask = np.ones(len(clean), bool)
    for p in cand:
        mask[max(0, p - int(0.20 * fs)):p + int(0.35 * fs)] = False
    big = np.flatnonzero(mask & (np.abs(clean) > 1.5 * amp_ref))
    if big.size:
        for g in np.split(big, np.where(np.diff(big) > int(0.3 * fs))[0] + 1):
            i = int(g[np.argmax(np.abs(clean[g]))])
            if not _in_any(t[i], removed):
                cand.setdefault(_snap(i, clean, fs), "大偏折")

    # 合并 250 ms 内的重复检测（保留斜率更大的），剔除既不陡也不大的候选（多为 T 波或噪声）
    beats = []
    for p in sorted(cand):
        s, a = _local_max(slope, p, int(0.06 * fs)), abs(clean[p])
        if s < 0.25 * slope_ref and a < 0.6 * amp_ref:
            continue
        if beats and p - beats[-1] < int(0.25 * fs):
            if s > _local_max(slope, beats[-1], int(0.06 * fs)):
                beats[-1] = p
            continue
        beats.append(p)

    # 高大 T 波：常规检测有时把它当成心搏（心率慢、T 波高时尤其如此），表现为每拍后面紧跟一个“提前搏动”。
    # 判据：距上一拍 < 0.42 s、斜率不到上一拍的 1/3，而且去掉它以后，前后两拍的间隔正好是一个正常 RR。
    # 阈值来自 MIT-BIH：被误检的 T 波斜率比 ≤ 0.27、距离 ≤ 0.35 s（第 90 百分位）；
    # 室性早搏斜率比 ≥ 0.35、距离 ≥ 0.45 s。插入性室性早搏若又缓又早，仍可能被误删。
    sl = np.array([_local_max(slope, p, int(0.06 * fs)) for p in beats])
    if len(beats) >= 5:
        strong = np.array(beats)[sl >= 0.5 * np.percentile(sl, 75)]
        rr_ref = float(np.median(np.diff(strong))) if len(strong) >= 3 else None
        keep = [True] * len(beats)
        if rr_ref:
            for i in range(1, len(beats) - 1):
                if not keep[i - 1]:
                    continue
                gap = beats[i + 1] - beats[i - 1]
                if (beats[i] - beats[i - 1] < int(T_GUARD_MAX_S * fs) and sl[i] < T_GUARD_SLOPE * sl[i - 1]
                        and abs(gap - rr_ref) <= 0.2 * rr_ref):
                    keep[i] = False
        # 首尾：记录在 T 波处开始或结束时没有“下一拍”可比，只看位置和斜率
        if beats[0] < int(T_GUARD_MAX_S * fs) and sl[0] < T_GUARD_SLOPE * sl[1]:
            keep[0] = False
        if beats[-1] - beats[-2] < int(T_GUARD_MAX_S * fs) and sl[-1] < T_GUARD_SLOPE * sl[-2] and keep[-2]:
            keep[-1] = False
        beats = [p for p, k in zip(beats, keep) if k]
    return beats, {p: cand[p] for p in beats}, clean, slope


# ----------------------------------------------------------------------------
# 形态
# ----------------------------------------------------------------------------
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or a.size != b.size:
        return 0.0
    a, b = detrend(a), detrend(b)
    sa, sb = a.std(), b.std()
    return 0.0 if sa == 0 or sb == 0 else float(np.mean(a * b) / (sa * sb))


def _qrs_bounds(sm: np.ndarray, center: int, fs: int, extend: bool = False) -> tuple[int, int]:
    """QRS 起点和 J 点：陡峭段斜率降到最大斜率 10% 处。峰顶谷底斜率为零，用约 18 ms 滑动最大值连成一片。

    extend=True 时再把紧邻的缓慢首末波算进去（只用于平均心搏的测量；逐拍宽度比较用基本规则，
    单拍噪声大，延伸规则在单拍上不稳定，会把正常心搏误判为“宽出”）。"""
    d = np.abs(np.gradient(sm)) * fs
    qa, qb = max(1, center - int(0.08 * fs)), min(len(sm) - 2, center + int(0.10 * fs))
    dmax = d[qa:qb].max() if qb > qa else 0.0
    if dmax <= 0:
        return center, center
    thr = 0.10 * dmax
    env = maximum_filter1d(d, size=max(3, int(0.018 * fs)))
    steep = np.flatnonzero(d[qa:qb] > 0.5 * dmax) + qa
    on, J = int(steep[0]), int(steep[-1])
    lo, hi = max(1, qa - int(0.06 * fs)), min(len(sm) - 2, qb + int(0.06 * fs))
    while on > lo and env[on - 1] > thr:
        on -= 1
    while d[on] <= thr and on < steep[0]:
        on += 1
    while J < hi and env[J + 1] > thr:
        J += 1
    while d[J] <= thr and J > steep[-1]:
        J -= 1
    if not extend:
        return on, J
    # 缓慢的首末波：例如右束支阻滞的宽 S 波，谷底附近斜率跌到阈值以下、随后又缓慢回升，
    # 上面的规则会把 J 点停在谷底。若 60 ms 内紧接着还有一段明显的偏折（斜率 ≥ 最大斜率的 15%，
    # 幅度 ≥ QRS 峰峰值的 20%），就把它算进 QRS，再按同一规则走到它结束。起点同理。
    p2p = float(sm[on:J + 1].max() - sm[on:J + 1].min())
    look = int(0.06 * fs)
    for _ in range(2):
        a, b = J + 1, min(len(sm) - 2, J + look, hi)
        if b <= a:
            break
        k = a + int(np.argmax(d[a:b]))
        e = k
        while e < hi and env[e + 1] > thr:
            e += 1
        if d[k] < 0.15 * dmax or abs(sm[e] - sm[J]) < 0.20 * p2p:
            break
        J = e
        while d[J] <= thr and J > k:
            J -= 1
    for _ in range(2):
        a, b = max(1, on - look, lo), on - 1
        if b <= a:
            break
        k = a + int(np.argmax(d[a:b]))
        s = k
        while s > lo and env[s - 1] > thr:
            s -= 1
        if d[k] < 0.15 * dmax or abs(sm[on] - sm[s]) < 0.20 * p2p:
            break
        on = s
        while d[on] <= thr and on < k:
            on += 1
    return on, J


def _tangent_tend(sm, d, tpk, base, fs, sign):
    """切线法 T 波终点：T 峰后最陡下降处的切线与基线的交点。"""
    seg = sign * d[tpk:tpk + int(0.18 * fs)]
    if seg.size < 3:
        return None
    k = tpk + int(np.argmin(seg))
    if d[k] == 0:
        return None
    return k + (base - sm[k]) / (d[k] / fs)


def beat_morphology(v, beats, fs, removed_flags):
    """以主导形态为模板：对齐（±50 ms）、QRS 与 QRS–T 相关系数、单拍 QRS 宽度。

    相关系数在去基线、轻度平滑（约 29 ms）后的信号上计算，减少基线漂移和肌电噪声的影响。
    单拍宽度与“形态一致的心搏的单拍宽度中位数”比较（同一规则、同样的噪声），而不是与平均模板比较。"""
    v_raw = v
    win = _odd(0.029 * fs)
    v = savgol_filter(v - _baseline(v, fs), win, 3)
    A, B = int(0.25 * fs), int(0.45 * fs)            # 模板窗口：R 前 0.25 s 到后 0.45 s
    qa, qb = int(0.10 * fs), int(0.12 * fs)          # QRS 窗口
    lag = int(0.05 * fs)                             # R、S 振幅接近时定位点会在两者间跳动（相隔约 25 ms），对齐要能覆盖

    def full(i):
        return i - A >= 0 and i + B <= len(v)

    def template(idx):
        segs = [v[i - A:i + B] for i in idx if full(i)]
        return np.median(np.array(segs), axis=0) if len(segs) >= 3 else None

    def align(i, tmpl):
        best, bi = -2.0, i
        tq = tmpl[A - qa:A + qb]
        for s in range(-lag, lag + 1):
            j = i + s
            if j - qa < 0 or j + qb > len(v):
                continue
            c = _corr(v[j - qa:j + qb], tq)
            if c > best:
                best, bi = c, j
        if best >= ODD_QRS_CORR:
            return bi, best
        if i - qa >= 0 and i + qb <= len(v):
            return i, _corr(v[i - qa:i + qb], tq)
        return i, None                                  # 记录首尾、QRS 窗口不完整：形态未评估

    base_idx = [b for b, r in zip(beats, removed_flags) if not r]
    tmpl = template(base_idx) if len(base_idx) >= 3 else None
    if tmpl is None:
        tmpl = template(beats)
    if tmpl is None:
        raise AnalysisError("完整的心搏少于 3 个，无法建立主导形态")
    pos, cq = zip(*[align(b, tmpl) for b in beats])
    good = [p for p, c, r in zip(pos, cq, removed_flags) if c is not None and c >= 0.9 and not r]
    tmpl2 = template(good)
    if tmpl2 is not None:
        tmpl = tmpl2
        pos, cq = zip(*[align(b, tmpl) for b in beats])
    pos, cq = list(pos), list(cq)

    cqt, widths = [], []
    for k, p in enumerate(pos):
        nxt = pos[k + 1] - p if k + 1 < len(pos) else B
        L = max(qb, min(B, int(0.85 * nxt)))
        if cq[k] is None:
            cqt.append(None)
        else:
            L = min(L, len(v) - p)
            cqt.append(_corr(v[p - qa:p + L], tmpl[A - qa:A + L]))
        a1, b1 = min(A, p), min(B, len(v) - p)
        seg = v_raw[p - a1:p + b1]
        if seg.size >= win + 2 and a1 >= int(0.1 * fs) and b1 >= int(0.18 * fs):
            on, J = _qrs_bounds(savgol_filter(seg, win, 3), a1, fs)
            widths.append((J - on) / fs * 1000)
        else:
            widths.append(None)
    ref = [w for w, c, r in zip(widths, cq, removed_flags) if w is not None and c is not None and c >= 0.9 and not r]
    if len(ref) < 3:
        ref = [w for w in widths if w is not None]
    width_t = float(np.median(ref)) if ref else 0.0
    return pos, cq, cqt, widths, width_t


def measure_template(W: np.ndarray, pre: int, rr_mean: float, fs: int) -> dict:
    """在平均心搏上测量 P、PR、QRS、ST、T、QT。W 为对齐好的单拍矩阵（已减去各拍基线）。"""
    win = _odd(0.029 * fs)
    b0 = pre - int(min(0.30, 0.45 * rr_mean) * fs)
    b1 = pre - int(min(0.21, 0.32 * rr_mean) * fs)
    W = W - W[:, b0:b1].mean(axis=1, keepdims=True)              # 逐拍减去 TP 段基线
    tm = np.median(W, axis=0)
    sm = savgol_filter(tm, win, 3)
    d = np.gradient(sm) * fs
    base_tp = float(np.mean(sm[b0:b1]))                          # TP 段：P 波的参考
    on, J = _qrs_bounds(sm, pre, fs, extend=True)
    # PQ 段（QRS 起点前 15–40 ms）：ST、T 波和 QT 的参考，也是 ST 测量的标准参考点。
    # 心率快时 TP 段很短，会混进上一拍的 T 波或本拍的 P 波；PQ 段不受影响。
    q0, q1 = on - int(0.040 * fs), on - int(0.015 * fs)
    base = float(np.mean(sm[q0:q1]))
    R = on + int(np.argmax(sm[on:J + 1]))
    S = R + int(np.argmin(sm[R:J + 1]))
    # 振幅逐拍取极值再取中位数：避免多拍对齐的微小抖动把窄 QRS 的峰值抹平
    r_amp = float(np.median(W[:, on:J + 1].max(axis=1)))
    s_amp = float(np.median(W[:, R:J + 1].min(axis=1)))

    ta = J + int(0.06 * fs)
    tb = min(pre + int(0.45 * fs), pre + int(0.70 * rr_mean * fs), len(sm) - int(0.2 * fs))
    Tpk = Tend = None
    t_sign, qts = 1, np.array([])
    if tb > ta + int(0.02 * fs):
        Tpk = ta + int(np.argmax(np.abs(sm[ta:tb] - base)))
        t_sign = 1 if sm[Tpk] >= base else -1
        Tend = _tangent_tend(sm, d, Tpk, base, fs, t_sign)
        q = []
        for w in W:                                             # 逐拍 QT（同一套规则），看稳定性
            s2 = savgol_filter(w, win, 3)
            d2 = np.gradient(s2) * fs
            b2 = float(np.mean(s2[q0:q1]))
            tp2 = ta + int(np.argmax(np.abs(s2[ta:tb] - b2)))
            te2 = _tangent_tend(s2, d2, tp2, b2, fs, t_sign)
            if te2:
                q.append((te2 - on) / fs * 1000)
        qts = np.array(q)
    qt = (Tend - on) / fs * 1000 if Tend else None

    # P 波：QRS 起点前 60 ms 到"250 ms 或 RR 的 35%"之间，正负两个方向都看
    # （起点前 60 ms 以内常有一段缓慢下凹，不算作 P 波）
    pa, pb = on - int(min(0.25, 0.35 * rr_mean) * fs), on - int(0.06 * fs)
    P, p_amp, p_pol, p_ok, pr = None, 0.0, None, False, None
    if pb > pa + 5:
        seg = sm[pa:pb] - base_tp
        ip, ineg = int(np.argmax(seg)), int(np.argmin(seg))
        if seg[ip] >= -seg[ineg]:
            P, p_amp, p_pol = pa + ip, float(seg[ip]), "直立"
        else:
            P, p_amp, p_pol = pa + ineg, float(seg[ineg]), "倒置"
        noise_p = float(np.std(W[:, pa:pb] - tm[pa:pb]))
        # 模板是多拍中位数，噪声约为单拍噪声 / √n
        p_ok = bool(abs(p_amp) > max(0.025, 3 * 1.25 * noise_p / np.sqrt(len(W))))
        if p_ok:
            po = P
            while po > pa - int(0.05 * fs) and abs(sm[po] - base_tp) > 0.15 * abs(p_amp):
                po -= 1
            pr = (on - po) / fs * 1000

    return dict(tm=tm, sm=sm, base=base, on=on, J=J, R=R, S=S, Tpk=Tpk, Tend=Tend, P=P,
                p_amp=p_amp, p_pol=p_pol if p_ok else None, p_ok=p_ok, pr=pr,
                qrs=(J - on) / fs * 1000, r_amp=r_amp, s_amp=s_amp,
                st_j=float(sm[J] - base), st_j60=float(sm[min(J + int(0.06 * fs), len(sm) - 1)] - base),
                t_amp=float(sm[Tpk] - base) if Tpk is not None else None,
                t_pol=("直立" if t_sign > 0 else "倒置") if Tpk is not None else None,
                qt=qt, qts=qts, noise=float(np.std(W - tm)), n=len(W), W=W)


# ----------------------------------------------------------------------------
# 形态提醒
# ----------------------------------------------------------------------------
def morphology_alerts(mo: dict | None, hr: float, sex: str = "unknown", placement: str = "wrist") -> tuple[list[dict], list[str]]:
    """根据平均心搏的测量值给出形态提醒。返回 (提醒列表, 跳过说明)。

    每条提醒：code、level（"关注" 或 "建议就医"）、value、text。
    这是单导联、约 30 秒记录上的筛查提示，不是诊断。"""
    alerts, skipped = [], []
    if mo is None:
        return alerts, ["没有可用的平均心搏，未做形态提醒"]
    if sex not in SEXES:
        raise ValueError(f"sex 只能是 {SEXES}")
    if placement not in PLACEMENTS:
        raise ValueError(f"placement 只能是 {PLACEMENTS}")
    few = mo["template_beats"] < MIN_TEMPLATE_BEATS
    tail = f"（平均心搏只有 {mo['template_beats']} 拍，可信度低）" if few else ""

    def add(code, level, value, text):
        alerts.append(dict(code=code, level=level, value=value, text=text + tail))

    # ---- QTc
    use_f = hr > QTC_FRIDERICIA_ABOVE_HR
    qtc = mo["qtc_fridericia_ms"] if use_f else mo["qtc_bazett_ms"]
    formula = "Fridericia" if use_f else "Bazett"
    iqr = mo.get("qt_beat_iqr_ms")
    t_amp = mo.get("t_amp_mv")
    if qtc is None:
        skipped.append("T 波终点未能测出，" + SKIP_QT)
    elif t_amp is None or abs(t_amp) < QT_MIN_T_MV:
        skipped.append(f"T 波太低（{t_amp} mV < {QT_MIN_T_MV}），T 终点不可靠，" + SKIP_QT)
    elif iqr and iqr[1] - iqr[0] > QT_MAX_IQR_MS:
        skipped.append(f"逐拍 QT 波动太大（四分位距 {iqr[1] - iqr[0]:.0f} ms），" + SKIP_QT)
    else:
        lim = QTC_LONG[sex]
        who = {"male": "男性", "female": "女性", "unknown": "性别未指定，按男性"}[sex]
        if qtc >= QTC_LONG_MARKED:
            add("QTC_LONG_MARKED", "建议就医", qtc,
                f"QTc 明显延长：{qtc:.0f} ms（{formula}，≥{QTC_LONG_MARKED}）。"
                "QTc ≥500 ms 与恶性室性心律失常风险显著相关；手表测得的 QT 通常还比 12 导联短。"
                "请尽快做 12 导联心电图，并带上正在服用的药物清单（不少药物会延长 QT）")
        elif qtc >= lim:
            add("QTC_LONG", "关注", qtc,
                f"QTc 偏长：{qtc:.0f} ms（{formula}，{who}阈值 {lim}）。"
                "单导联测量有误差；若反复出现，或有晕厥、心悸、家族猝死史，或在服用可能延长 QT 的药物，请做 12 导联心电图")
        elif qtc <= QTC_SHORT_MARKED:
            add("QTC_SHORT_MARKED", "建议就医", qtc,
                f"QTc 明显偏短：{qtc:.0f} ms（{formula}，≤{QTC_SHORT_MARKED}）。"
                "手表的 QT 常比 12 导联短 10–25 ms，且 T 波终点在单导联上容易判早；请用 12 导联心电图确认")
        elif qtc <= QTC_SHORT:
            add("QTC_SHORT", "关注", qtc,
                f"QTc 偏短：{qtc:.0f} ms（{formula}，≤{QTC_SHORT}）。"
                "手表的 QT 常比 12 导联短 10–25 ms，这个值本身不足以说明问题；"
                "若有不明原因晕厥、心悸或家族猝死史，请做 12 导联心电图确认")

    # ---- QRS 宽度
    if mo["qrs_ms"] is not None and mo["qrs_ms"] >= QRS_WIDE_MS:
        add("QRS_WIDE", "关注", mo["qrs_ms"],
            f"QRS 增宽：{mo['qrs_ms']:.0f} ms（≥{QRS_WIDE_MS}）。可能是束支传导阻滞或室内传导延迟，"
            "也可能是单导联上测量的误差；以前没有发现过的话，请做 12 导联心电图")

    # ---- PR
    if not mo["p_measurable"] or mo["pr_ms"] is None:
        skipped.append("P 波不可测，未做 PR 提醒")
    elif hr > PR_MAX_HR:
        skipped.append(f"心率 {hr:.0f} > {PR_MAX_HR}：P 波常与上一拍的 T 波重叠，PR 不可靠，未做 PR 提醒")
    elif mo["pr_ms"] > PR_LONG_MS:
        add("PR_LONG", "关注", mo["pr_ms"],
            f"PR 间期延长：{mo['pr_ms']:.0f} ms（>{PR_LONG_MS}）。可能是一度房室传导阻滞（运动员和年轻人中较常见，"
            "多数无害），也可能是 P 波起点判早；请在 12 导联心电图上确认")
    elif mo["pr_ms"] < PR_SHORT_MS:
        add("PR_SHORT", "关注", mo["pr_ms"],
            f"PR 间期偏短：{mo['pr_ms']:.0f} ms（<{PR_SHORT_MS}）。可能是正常变异，也可能是预激（如 WPW）；"
            "有阵发性心动过速的话，请做 12 导联心电图")

    # ---- ST、T：依赖导联方向，只对手腕的标准记录方式做判断
    if placement != "wrist":
        skipped.append("记录时手表不在手腕上（非标准导联），未做 ST 和 T 波提醒")
    elif any(a["code"] == "QRS_WIDE" for a in alerts):
        skipped.append("QRS 增宽时 ST 段和 T 波的改变多为继发性，未单独做 ST 和 T 波提醒")
    else:
        st_j, st_60 = mo["st_j_mv"], mo["st_j60_mv"]
        if st_j is not None and st_j >= ST_DEV_MV:
            add("ST_ELEVATION", "关注", st_j,
                f"ST 段抬高：J 点 +{st_j:.2f} mV（≥{ST_DEV_MV}）。单导联不能诊断；"
                "若伴胸痛、胸闷、出冷汗，请立即就医；无症状时也可能是早复极等正常变异")
        elif st_j is not None and st_60 is not None and st_j <= -ST_DEV_MV and st_60 <= -ST_DEV_MV:
            add("ST_DEPRESSION", "关注", st_j,
                f"ST 段压低：J 点 {st_j:.2f} mV、J 点后 60 ms {st_60:.2f} mV（均 ≤-{ST_DEV_MV}，呈水平或下斜型）。"
                "单导联不能诊断；宽而缓的 S 波（如右束支阻滞）的末段也可能被算作 ST 段。若伴胸痛、胸闷，请立即就医")
        if (mo["t_polarity"] == "倒置" and t_amp is not None and abs(t_amp) >= T_INV_MIN_MV
                and mo["r_amp_mv"] > abs(mo["s_amp_mv"])):
            add("T_INVERSION", "关注", t_amp,
                f"T 波倒置：{t_amp:.2f} mV，而 QRS 主波向上。导联 I 中 T 波通常直立；"
                "请在 12 导联心电图上确认")
    return alerts, skipped


# ----------------------------------------------------------------------------
# 分析主流程
# ----------------------------------------------------------------------------
def analyze(rec: Recording, exclude: list[tuple[float, float]] | None = None,
            sex: str = "unknown", placement: str = "wrist") -> dict:
    fs, t, v = rec.fs, rec.t, rec.v
    exclude = [(float(a), float(b)) for a, b in (exclude or [])]
    artifacts = _artifact_windows(t, v, fs)
    removed = exclude + artifacts

    beats, sources, clean, _ = detect_beats(t, v, fs, removed)
    rem_flags = [_in_any(t[b], removed) for b in beats]
    pos, cq, cqt, widths, width_t = beat_morphology(v, beats, fs, rem_flags)
    # 记录开头的不完整波形：开头那一拍的 QRS 落在记录开始之前，紧随其后的 T 波可能被当成心搏
    notes = list(rec.notes)                         # 不改动 rec，同一条记录可以重复分析
    head = [k for k in range(len(pos)) if t[pos[k]] - t[0] < 0.45 and cq[k] is not None and cq[k] < 0.5]
    for k in reversed(head):
        notes.append(f"记录开头 {t[pos[k]]:.2f} 秒处的不完整波形已忽略（多为记录开始前那一拍的 T 波）")
        for lst in (beats, rem_flags, pos, cq, cqt, widths):
            del lst[k]
    times = t[np.array(pos)]
    nb = len(pos)
    if nb < 5:
        raise AnalysisError("可用心搏少于 5 个，无法分析")

    # ---- RR：相邻两拍之间的间隔；跨越排除段的间隔不可用（也不会被当作停顿）
    rr = np.diff(times)
    usable = np.array([not (rem_flags[k] or rem_flags[k + 1] or _overlaps(times[k], times[k + 1], removed))
                       and rr[k] > 0.25 for k in range(nb - 1)], bool)
    if usable.sum() < 3:
        raise AnalysisError("可用的 RR 间期少于 3 个（伪影或排除段太多）")
    glob_med = float(np.median(rr[usable]))

    def ref_for(skip: set[int], center: int) -> float:
        """局部参考 RR：邻近间期两两取平均后再取中位数。
        相邻两个间期的平均值能抵消长短交替、早搏与代偿间歇的影响，中位数再挡住个别异常值。"""
        js = [j for j in range(max(0, center - 6), min(len(rr), center + 7)) if usable[j] and j not in skip]
        pairs = [(rr[a] + rr[b]) / 2 for a, b in zip(js, js[1:]) if b == a + 1]
        if len(pairs) >= 3:
            return float(np.median(pairs))
        return float(np.median([rr[j] for j in js])) if len(js) >= 3 else glob_med

    # ---- 逐拍标注
    labels, prem_ratio, comp_ratio = [], [], []
    for k in range(nb):
        if rem_flags[k]:
            labels.append(LABEL_REMOVED); prem_ratio.append(None); comp_ratio.append(None)
            continue
        pr_ = cp_ = None
        if k >= 1 and usable[k - 1]:
            ref = ref_for({k - 1, k}, k - 1)
            pr_ = rr[k - 1] / ref
            if k < nb - 1 and usable[k]:
                cp_ = (rr[k - 1] + rr[k]) / (2 * ref)
        wide = widths[k] is not None and widths[k] - width_t >= WIDE_QRS_DELTA_MS
        c = cq[k] if cq[k] is not None else 1.0         # 形态未评估的首尾心搏只按时间判断
        abrupt = True
        if k >= 2 and usable[k - 2] and labels[k - 1] not in (LABEL_PREM_SAME, LABEL_PREM_DIFF):
            abrupt = rr[k - 1] < ABRUPT_RATIO * rr[k - 2]   # 连续早搏的第二拍不做此检查
        if pr_ is not None and pr_ < PREMATURE_RATIO and abrupt:
            labels.append(LABEL_PREM_DIFF if (c < SAME_QRS_CORR or wide) else LABEL_PREM_SAME)
        elif c < ODD_QRS_CORR or wide:
            labels.append(LABEL_ODD)
        else:
            labels.append(LABEL_NORMAL)
        prem_ratio.append(pr_); comp_ratio.append(cp_)

    # ---- 长间歇（早搏之后的长间隔算代偿间歇，不单独报）
    pauses, comp_pauses = [], []
    for k in range(nb - 1):
        if not usable[k]:
            continue
        ratio = rr[k] / ref_for({k}, k)
        if ratio >= PAUSE_RATIO:
            item = dict(start_s=round(float(times[k]), 2), end_s=round(float(times[k + 1]), 2),
                        rr_ms=round(float(rr[k] * 1000)), ratio=round(float(ratio), 2))
            (comp_pauses if labels[k] in (LABEL_PREM_SAME, LABEL_PREM_DIFF) else pauses).append(item)
    pause_starts = {p["start_s"] for p in pauses}

    # ---- NN 间期：两端都是正常心搏、可用、不是长间歇
    nn_idx = [k for k in range(nb - 1)
              if usable[k] and labels[k] == LABEL_NORMAL and labels[k + 1] == LABEL_NORMAL
              and 0.3 < rr[k] < 2.0 and round(float(times[k]), 2) not in pause_starts]
    nn = rr[nn_idx] * 1000
    succ = np.array([nn[j + 1] - nn[j] for j in range(len(nn_idx) - 1) if nn_idx[j + 1] == nn_idx[j] + 1])

    # ---- 长短交替（用全部可用间期）：连续 ≥4 次方向交替且每次变化 > 80 ms
    alt_runs = []
    us_k = [k for k in range(len(rr)) if usable[k]]
    run, start = 0, None
    for a, b, c in zip(us_k, us_k[1:], us_k[2:]):
        if b != a + 1 or c != b + 1:
            run = 0
            continue
        d1, d2 = (rr[b] - rr[a]) * 1000, (rr[c] - rr[b]) * 1000
        if np.sign(d1) == -np.sign(d2) and abs(d1) > ALT_MIN_DELTA_MS and abs(d2) > ALT_MIN_DELTA_MS:
            if run == 0:
                start = a
            run += 1
            if run == 3:
                alt_runs.append(round(float(times[start]), 1))
        else:
            run = 0

    # ---- 二联律样：≥3 个提前搏动、彼此正好隔一拍
    prem_k = [k for k, l in enumerate(labels) if l in (LABEL_PREM_SAME, LABEL_PREM_DIFF)]
    bigeminy = None
    run_k = [prem_k[0]] if prem_k else []
    for a, b in zip(prem_k, prem_k[1:]):
        run_k = run_k + [b] if b - a == 2 else [b]
        if len(run_k) >= 3:
            bigeminy = round(float(times[run_k[0]]), 1)

    # ---- 心率、HRV
    hr = 60.0 / float(np.mean(rr[usable]))
    hr_nn = 60.0 / float(np.mean(nn) / 1000) if nn.size else None
    sdnn = float(np.std(nn, ddof=1)) if nn.size > 2 else None
    rmssd = float(np.sqrt(np.mean(succ ** 2))) if succ.size else None
    pnn50 = float(np.mean(np.abs(succ) > 50) * 100) if succ.size else None
    sd1 = float(np.sqrt(0.5 * np.var(succ, ddof=1))) if succ.size > 2 else None
    sd2 = float(np.sqrt(max(2 * sdnn ** 2 - 0.5 * np.var(succ, ddof=1), 0))) if (succ.size > 2 and sdnn) else None

    # ---- 平均心搏形态（未经高通滤波的原始信号）
    rr_mean = float(np.mean(nn) / 1000) if nn.size else glob_med
    pre, post = int(min(0.40, 0.75 * rr_mean) * fs), int(min(0.60, 0.92 * rr_mean) * fs)
    morph = None
    for thr in (TEMPLATE_CORR, 0.80):
        tb_idx = [k for k in range(nb) if labels[k] == LABEL_NORMAL and cqt[k] is not None and cqt[k] >= thr
                  and pos[k] - pre >= 0 and pos[k] + post <= len(v)
                  and (k + 1 >= nb or pos[k + 1] - pos[k] >= post)]      # 窗口里不能带进下一拍（例如紧随的早搏）
        if len(tb_idx) >= 3:
            W = np.array([v[pos[k] - pre:pos[k] + post] for k in tb_idx])
            morph = measure_template(W, pre, rr_mean, fs)
            morph["beat_idx"] = tb_idx
            morph["pre"] = pre
            morph["corr_threshold"] = thr
            break

    clipped = bool(morph and any(rec.clip[max(0, pos[k] - pre):pos[k] + post].any() for k in morph["beat_idx"]))

    # ---- 提示
    flags = []
    watch = rec.watch
    t_list = lambda ks: "、".join(f"{times[k]:.1f}" for k in ks) + " 秒"
    same = [k for k in prem_k if labels[k] == LABEL_PREM_SAME]
    diff = [k for k in prem_k if labels[k] == LABEL_PREM_DIFF]
    odd = [k for k, l in enumerate(labels) if l == LABEL_ODD]
    if diff:
        flags.append(f"{len(diff)} 个提前搏动（QRS 异形）：{t_list(diff)}。可能为室性早搏（或伴差异传导的房性早搏），"
                     "请就医时提供原始 PDF")
    if same:
        flags.append(f"{len(same)} 个提前搏动（QRS 同形）：{t_list(same)}。可能为房性早搏，也可能是窦性心律不齐，单导联无法确定")
    if bigeminy is not None:
        flags.append(f"提前搏动从约 {bigeminy} 秒起规律地每隔一拍出现（二联律样）")
    if pauses:
        flags.append("长间歇：" + "；".join(f"{p['start_s']}–{p['end_s']} 秒（{p['rr_ms']} ms，约为参考间期的 {p['ratio']} 倍）"
                                            for p in pauses))
    if alt_runs:
        flags.append("RR 间期出现规律的长短交替（起点约 " + "、".join(f"{x} 秒" for x in alt_runs) +
                     "）：可能是呼吸相关的窦性心律不齐，也可能是房性二联律，单导联无法区分；此时 RMSSD 与 Poincaré 图不能按常规 HRV 解读")
    if odd:
        flags.append(f"{len(odd)} 个心搏出现时间正常但形态异常：{t_list(odd)}。多为伪影或接触变化，也可能是异常搏动，请人工查看")
    if removed:
        kinds = (["手动排除"] if exclude else []) + (["自动识别的伪影"] if artifacts else [])
        flags.append(f"以下时段被排除（{'、'.join(kinds)}）：" + "，".join(f"{a:.1f}–{b:.1f} 秒" for a, b in removed) +
                     "。跨越这些时段的间隔不计入 RR/HRV，也不会被当作停顿")
    if clipped:
        flags.append("波形碰到网格边缘被截断：R/S 振幅只是下限")
    if morph is None:
        flags.append("形态一致的正常心搏少于 3 个，未测量平均心搏")
    elif not morph["p_ok"]:
        flags.append("P 波低于可测阈值，PR 间期无法测量")
    rep_bpm = watch.get("reported_bpm")
    if rep_bpm and abs(hr - rep_bpm) > 3:
        flags.append(f"计算心率 {hr:.0f} 与手表报告 {rep_bpm} 相差超过 3 次/分，请检查提取结果")
    cat = watch.get("category")
    if watch.get("algorithm_version") == 2 and cat in WATCH_RANGES_V2:
        lo, hi = WATCH_RANGES_V2[cat]
        if not (lo - 2 <= hr <= hi + 2):
            flags.append(f"手表分类“{watch['classification']}”对应 {lo}–{hi} 次/分，与计算心率 {hr:.0f} 不一致，请检查提取结果")
    elif watch.get("algorithm_version") not in (None, 2):
        flags.append(f"手表算法版本为 {watch['algorithm_version']}，未内置其分类规则，跳过分类一致性检查")
    if cat in ("afib", "afib_high"):
        flags.append("手表提示房颤：本工具不做房颤判断，请以手表结果和医生意见为准")
    if cat == "sinus" and (prem_k or pauses):
        flags.append("手表判为窦性心律与上述提示并不矛盾：Apple 的算法只区分房颤与窦性心律，不识别早搏和停顿")
    age = watch.get("age_at_recording")
    if age is not None and age < APPLE_MIN_AGE:
        flags.append(f"记录时 {age} 岁：Apple 说明书注明心电图 App 不适用于 {APPLE_MIN_AGE} 岁以下人群，手表分类仅供参考（波形仍可分析）")
    if watch.get("symptoms"):
        flags.append(f"记录时补录的症状：{watch['symptoms']}")
    flags.append("HRV 来自约 30 秒的超短时记录，仅供同条件下自身对比")
    flags.append("PDF 一律标注“导联 I”；若手表不是贴在对侧手腕上记录，则不是标准导联 I")

    r1 = lambda x, n=1: None if x is None else round(float(x), n)
    m = morph or {}
    qt = m.get("qt")
    beats_table = [dict(time_s=r1(times[k], 3), label=labels[k], source=sources[beats[k]],
                        rr_prev_ms=r1(rr[k - 1] * 1000, 0) if k >= 1 else None,
                        rr_next_ms=r1(rr[k] * 1000, 0) if k < nb - 1 else None,
                        prematurity=r1(prem_ratio[k], 2), compensation=r1(comp_ratio[k], 2),
                        corr_qrs=r1(cq[k], 3), corr_qrst=r1(cqt[k], 3), qrs_ms=r1(widths[k], 0))
                   for k in range(nb)]
    report = {
        "tool_version": __version__,
        "record_id": rec.record_id,
        "source": rec.source,
        "watch": watch,
        "rhythm": {
            "beats": nb, "hr_bpm": r1(hr), "hr_nn_bpm": r1(hr_nn), "reported_bpm": rep_bpm,
            "rr_ms_min": r1(rr[usable].min() * 1000, 0), "rr_ms_max": r1(rr[usable].max() * 1000, 0),
            "n_normal": labels.count(LABEL_NORMAL), "n_premature_same": len(same), "n_premature_diff": len(diff),
            "n_odd": len(odd), "n_removed": labels.count(LABEL_REMOVED),
            "pauses": pauses, "compensatory_pauses": comp_pauses,
            "bigeminy_start_s": bigeminy, "rr_alternation_start_s": alt_runs,
        },
        "hrv": {"n_nn": int(nn.size), "mean_nn_ms": r1(np.mean(nn)) if nn.size else None,
                "sdnn_ms": r1(sdnn), "rmssd_ms": r1(rmssd), "pnn50_pct": r1(pnn50), "sd1_ms": r1(sd1), "sd2_ms": r1(sd2)},
        "morphology": None if not morph else {
            "template_beats": m["n"], "template_corr_threshold": m["corr_threshold"],
            "p_amp_mv": r1(m["p_amp"], 3), "p_polarity": m["p_pol"], "p_measurable": m["p_ok"], "pr_ms": r1(m["pr"], 0),
            "qrs_ms": r1(m["qrs"], 0), "r_amp_mv": r1(m["r_amp"], 3), "s_amp_mv": r1(m["s_amp"], 3),
            "amplitude_is_lower_bound": clipped, "st_j_mv": r1(m["st_j"], 3), "st_j60_mv": r1(m["st_j60"], 3),
            "t_amp_mv": r1(m["t_amp"], 3), "t_polarity": m["t_pol"], "qt_ms": r1(qt, 0),
            "qt_beat_iqr_ms": [r1(np.percentile(m["qts"], 25), 0), r1(np.percentile(m["qts"], 75), 0)] if m["qts"].size else None,
            "qtc_bazett_ms": r1(qt / np.sqrt(rr_mean), 0) if qt else None,
            "qtc_fridericia_ms": r1(qt / np.cbrt(rr_mean), 0) if qt else None,
        },
        "quality": {"noise_sd_mv": r1(m.get("noise"), 3), "clipped": clipped,
                    "artifact_windows_s": [[r1(a), r1(b)] for a, b in artifacts], "manual_exclude_s": exclude},
        "flags": flags,
        "notes": notes,
        "beats": beats_table,
    }
    alerts, skipped = morphology_alerts(report["morphology"], hr, sex, placement)
    report["morphology_alerts"] = {"sex": sex, "placement": placement, "alerts": alerts, "skipped": skipped}
    if report["morphology"] is not None:
        report["morphology"]["qt_reliable"] = not any(x.endswith(SKIP_QT) for x in skipped)
    # 形态提醒放在最前面（“建议就医”优先）
    alerts_sorted = sorted(alerts, key=lambda a: a["level"] != "建议就医")
    report["flags"] = [f"【{a['level']}】{a['text']}" for a in alerts_sorted] + flags
    if placement != "wrist":
        report["flags"].append("已声明手表不在手腕上记录：波形方向和幅度与标准导联 I 不可比")
    return dict(report=report, times=times, pos=pos, labels=labels, removed=removed, morph=morph,
                nn=nn, pauses=pauses)


# ----------------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------------
MARKERS = {
    LABEL_NORMAL: dict(marker="o", color="#1565c0", s=14),
    LABEL_PREM_SAME: dict(marker="^", color="#ef6c00", s=70),
    LABEL_PREM_DIFF: dict(marker="X", color="#c62828", s=90),
    LABEL_ODD: dict(marker="s", color="#616161", s=45),
    LABEL_REMOVED: dict(marker="o", color="#bdbdbd", s=14),
}


def dashboard_title(report: dict) -> str:
    w = report["watch"]
    parts = [report["record_id"], w.get("recorded_at") or "", w.get("classification") or "",
             f"手表 {w['reported_bpm']} 次/分" if w.get("reported_bpm") else ""]
    if w.get("symptoms"):
        parts.append(f"症状：{w['symptoms']}")
    return "  ".join(p for p in parts if p)


def export(rec: Recording, res: dict, out_dir: Path, include_identity: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rec.record_id
    rep = dict(res["report"])
    if include_identity:
        rep["identity"] = rec.identity
    fs = rec.fs

    # CSV：连续信号 + 心搏标注
    lab = np.array([""] * len(rec.v), dtype=object)
    for p, l in zip(res["pos"], res["labels"]):
        lab[p] = l
    pd.DataFrame({"time_s": rec.t.round(5), "voltage_mv": rec.v.round(5),
                  "clipped": rec.clip.astype(int), "beat": lab}).to_csv(out_dir / f"{stem}_signal.csv", index=False,
                                                                       encoding="utf-8-sig")
    with open(out_dir / f"{stem}_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # 看板
    n_rows = max(r.index for r in rec.rows) + 1                       # PDF 里实际的波形行数
    fig = plt.figure(figsize=(16, 2.8 * n_rows + 5.5), dpi=130)
    gs = fig.add_gridspec(n_rows + 1, 3, height_ratios=[1] * n_rows + [2.2])
    times, labels = res["times"], res["labels"]
    for k in range(n_rows):
        ax = fig.add_subplot(gs[k, :])
        t0, t1 = k * STRIP_SECONDS, (k + 1) * STRIP_SECONDS
        msk = (rec.t >= t0) & (rec.t < t1)
        ax.plot(rec.t[msk], rec.v[msk], color="#c62828", lw=0.8)
        for a, b in res["removed"]:
            if b >= t0 and a <= t1:
                ax.axvspan(max(a, t0), min(b, t1), color="#999", alpha=0.25)
        for lname, st in MARKERS.items():
            sel = [i for i, l in enumerate(labels) if l == lname and t0 <= times[i] < t1]
            if sel:
                ax.scatter(times[sel], rec.v[np.array(res["pos"])[sel]], zorder=3, **st)
            if lname not in (LABEL_NORMAL, LABEL_REMOVED):
                for i in sel:
                    ax.annotate(f"{times[i]:.2f}s", (times[i], rec.v[res["pos"][i]]), (0, 10),
                                textcoords="offset points", ha="center", fontsize=8, color=st["color"])
        top = rec.v[msk].max() if msk.any() else 1
        for p in res["pauses"]:
            if t0 <= p["start_s"] < t1:
                ax.annotate("", (p["start_s"], top), (p["end_s"], top),
                            arrowprops=dict(arrowstyle="|-|", color="#c62828", lw=1))
                ax.text((p["start_s"] + p["end_s"]) / 2, top, f"长间歇 {p['rr_ms']} ms", ha="center",
                        va="bottom", fontsize=8, color="#c62828")
        ax.set_xlim(t0, t1); ax.set_ylabel("mV"); ax.grid(alpha=0.25)
        if rec.clip[msk].any():
            ax.text(0.995, 0.92, "截断", transform=ax.transAxes, ha="right", color="#b71c1c", fontsize=9)
        if k == 0:
            ax.set_title(dashboard_title(rep), loc="left", fontsize=11)
            handles = [Line2D([], [], ls="", markersize=7, marker=st["marker"], color=st["color"], label=l)
                       for l, st in MARKERS.items()] + \
                      [plt.Rectangle((0, 0), 1, 1, color="#999", alpha=0.25, label="排除时段")]
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.5, ncol=6,
                      frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.axes[-1].set_xlabel("时间（s）")

    m = res["morph"]
    ax = fig.add_subplot(gs[n_rows, :2])
    mo = rep["morphology"]
    if m:
        base, R = m["base"], m["R"]
        tt = (np.arange(len(m["tm"])) - R) / fs * 1000
        for w in m["W"]:
            ax.plot(tt, w - base, color="#f2c4c4", lw=0.5)
        sm = m["sm"] - base
        ax.plot(tt, sm, color="#c62828", lw=2)
        ax.axhline(0, color="#888", lw=0.6, ls="--")

        def lab_(i, txt, dy, dx=0.0, ha="center"):
            ax.annotate(txt, (tt[i], sm[i]), (tt[i] + dx, sm[i] + dy), ha=ha, va="center", fontsize=9,
                        arrowprops=dict(arrowstyle="-", color="#555", lw=0.6))
        if m["P"] is not None:
            lab_(m["P"], f"P {mo['p_amp_mv']:.2f} mV" + ("" if m["p_ok"] else "（不可测）"), 0.3)
        lab_(R, f"R {mo['r_amp_mv']:.2f} mV", 0.15)
        lab_(m["S"], f"S {'≤ ' if mo['amplitude_is_lower_bound'] else ''}{mo['s_amp_mv']:.2f} mV", 0.0, dx=70, ha="left")
        if m["Tpk"] is not None:
            lab_(m["Tpk"], f"T {mo['t_amp_mv']:.2f} mV", 0.25)
        lo = min(sm.min(), -0.5)
        ax.annotate("", (tt[m["on"]], lo - 0.15), (tt[m["J"]], lo - 0.15), arrowprops=dict(arrowstyle="<->", color="#1565c0"))
        ax.text((tt[m["on"]] + tt[m["J"]]) / 2, lo - 0.2, f"QRS {mo['qrs_ms']:.0f} ms", ha="center", va="top",
                color="#1565c0", fontsize=9)
        if m["Tend"]:
            te = (m["Tend"] - R) / fs * 1000
            ax.annotate("", (tt[m["on"]], lo - 0.45), (te, lo - 0.45), arrowprops=dict(arrowstyle="<->", color="#2e7d32"))
            ax.text((tt[m["on"]] + te) / 2, lo - 0.5,
                    f"QT {mo['qt_ms']:.0f} ms" + ("" if mo.get("qt_reliable", True) else "（不可靠）"), ha="center", va="top",
                    color="#2e7d32", fontsize=9)
        top = max(sm.max(), 0.5) + 0.5
        ax.set_ylim(lo - 0.75, top)
        keys = [("P", m["P"]), ("QRS起", m["on"]), ("R", R), ("S", m["S"]), ("J", m["J"]), ("T峰", m["Tpk"]), ("T末", m["Tend"])]
        last_x, level = -1e9, 0
        for name, i in sorted([k for k in keys if k[1] is not None], key=lambda k: k[1]):
            x = (i - R) / fs * 1000
            level = (level + 1) % 2 if x - last_x < 30 else 0            # 相距太近的标签错开一行
            last_x = x
            ax.axvline(x, color="#9e9e9e", lw=0.6, ls=(0, (3, 3)), alpha=0.35, zorder=0)
            ax.text(x, top - 0.05 - level * 0.13, name, ha="center", va="top", fontsize=7.5, color="#9e9e9e")
        ax.set_title(f"平均心搏（{m['n']} 拍：浅色 = 单拍，红线 = 中位数模板）", loc="left", fontsize=11)
    else:
        ax.text(0.5, 0.5, "形态一致的正常心搏不足，未测量平均心搏", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("距 R 峰的时间（ms）"); ax.set_ylabel("mV"); ax.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[n_rows, 2])
    nn = res["nn"]
    if nn.size > 2:
        ax2.scatter(nn[:-1], nn[1:], s=18, color="#6a1b9a", alpha=0.7)
        lo2, hi2 = nn.min() - 30, nn.max() + 30
        ax2.plot([lo2, hi2], [lo2, hi2], color="#999", ls="--", lw=0.7)
        ax2.set_xlim(lo2, hi2); ax2.set_ylim(lo2, hi2)
    ax2.set_xlabel("NN$_n$（ms）"); ax2.set_ylabel("NN$_{n+1}$（ms）"); ax2.grid(alpha=0.25)   # Unicode 下标在微软雅黑里缺字
    rh, hv = rep["rhythm"], rep["hrv"]
    fmt = lambda x, u="": "—" if x is None else f"{x}{u}"
    lines_ = [f"心率 {rh['hr_bpm']} 次/分（RR {rh['rr_ms_min']:.0f}–{rh['rr_ms_max']:.0f} ms）",
              f"提前搏动 同形 {rh['n_premature_same']} / 异形 {rh['n_premature_diff']}  形态异常 {rh['n_odd']}  长间歇 {len(rh['pauses'])}",
              f"RMSSD {fmt(hv['rmssd_ms'])}  SDNN {fmt(hv['sdnn_ms'])}  pNN50 {fmt(hv['pnn50_pct'], '%')}"]
    if mo:
        qtc_txt = (f"QTc {fmt(mo['qtc_bazett_ms'])}（Bazett）/ {fmt(mo['qtc_fridericia_ms'])}（Fridericia）"
                   if mo.get("qt_reliable", True) else "QTc 不可靠（T 波太低或逐拍波动大）")
        lines_.append(f"{qtc_txt}  PR {mo['pr_ms'] if mo['pr_ms'] else '不可测'}")
    al = rep.get("morphology_alerts", {}).get("alerts", [])
    lines_.append("形态提醒：" + ("、".join(ALERT_NAMES[a["code"]] for a in al) if al else "无"))
    ax2.set_title("\n".join(lines_), loc="left", fontsize=9,
                  color="#b71c1c" if any(a["level"] == "建议就医" for a in al) else "black")

    footer = "由 Apple Watch 导出文件还原，仅供个人记录与对比，非医疗诊断。完整提示见同名 _report.json"
    if rep["flags"]:
        first = rep["flags"][0]
        footer += "。首条提示：" + (first if len(first) <= 70 else first[:69] + "…")
    fig.text(0.01, 0.003, footer, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    fig.savefig(out_dir / f"{stem}_dashboard.png")
    plt.close(fig)
    return rep


def summary_row(rep: dict) -> dict:
    w, rh, hv, mo = rep["watch"], rep["rhythm"], rep["hrv"], rep["morphology"] or {}
    hour = int(w["recorded_at"][11:13]) if w.get("recorded_at") else None
    row = {"记录": rep["record_id"], "记录时间": w.get("recorded_at"),
           "时段": None if hour is None else ("夜间" if hour < 6 else "白天" if hour < 18 else "晚上"),
           "记录时年龄": w.get("age_at_recording"), "手表结论": w.get("classification"), "手表心率": w.get("reported_bpm"),
           "症状": w.get("symptoms"), "计算心率": rh["hr_bpm"], "心搏": rh["beats"],
           "提前_同形": rh["n_premature_same"], "提前_异形": rh["n_premature_diff"], "形态异常": rh["n_odd"],
           "长间歇": len(rh["pauses"]), "长短交替": bool(rh["rr_alternation_start_s"]),
           "QRS_ms": mo.get("qrs_ms"), "PR_ms": mo.get("pr_ms"), "QT_ms": mo.get("qt_ms"),
           "QTcB_ms": mo.get("qtc_bazett_ms"), "QTcF_ms": mo.get("qtc_fridericia_ms"), "QT可靠": mo.get("qt_reliable"),
           "P_mV": mo.get("p_amp_mv"), "R_mV": mo.get("r_amp_mv"), "S_mV": mo.get("s_amp_mv"), "T_mV": mo.get("t_amp_mv"),
           "RMSSD_ms": hv["rmssd_ms"], "SDNN_ms": hv["sdnn_ms"], "NN数": hv["n_nn"],
           "截断": rep["quality"]["clipped"],
           "形态提醒": "、".join(ALERT_NAMES[a["code"]] for a in rep.get("morphology_alerts", {}).get("alerts", [])),
           "提示数": len(rep["flags"])}
    if "identity" in rep:
        row["源文件"] = rep["identity"].get("source_file")
    return row


def _parse_excl(spec: str | None) -> list[tuple[float, float]]:
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):                     # 中文 Windows 重定向输出时避免编码错误
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Apple Watch 心电图提取与分析")
    ap.add_argument("inputs", nargs="+", help="PDF / 图片文件或文件夹")
    ap.add_argument("-o", "--output", default="./ecg_output")
    ap.add_argument("--exclude", help="手动排除的时间段（秒），如 0-4,27.5-30；只能用于单个文件")
    ap.add_argument("--include-identity", action="store_true", help="在报告中保留姓名、出生日期和源文件名")
    ap.add_argument("--sex", choices=SEXES, default="unknown",
                    help="QTc 延长的阈值按性别（男 450 / 女 460 ms）；未指定时按男性的较低阈值")
    ap.add_argument("--placement", choices=PLACEMENTS, default="wrist",
                    help="记录时手表的位置：wrist = 戴在手腕（标准方式）；other = 贴在胸口、腹部等（不做 ST/T 提醒）")
    args = ap.parse_args(argv)

    files = []
    for s in args.inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})
        else:
            files.append(p)
    if not files:
        print("[!] 没有找到 PDF 或图片文件")
        return 2
    if args.exclude and len(files) != 1:
        print("[!] --exclude 只能用于单个文件（批量处理时会误伤其他记录）")
        return 2
    try:
        excl = _parse_excl(args.exclude)
    except ValueError:
        print(f"[!] --exclude 格式应为 起-止[,起-止...]（秒），例如 0-4,27.5-30；收到：{args.exclude}")
        return 2
    out = Path(args.output)
    rows, used_ids, failed = [], set(), 0
    for n, f in enumerate(files, 1):
        try:
            rec = load(f)
            base_id, k = rec.record_id, 2
            while rec.record_id in used_ids:                       # 同一分钟的多条记录
                rec.record_id = f"{base_id}_{k}"; k += 1
            used_ids.add(rec.record_id)
            print(f"\n>>> [{n}/{len(files)}] {rec.record_id}" + (f"（{f.name}）" if args.include_identity else ""))
            res = analyze(rec, excl, sex=args.sex, placement=args.placement)
            rep = export(rec, res, out, args.include_identity)
        except Exception as e:                                     # 单个文件失败不影响批量
            failed += 1
            msg = str(e)
            if not args.include_identity:                          # 异常信息里可能带完整路径（文件名含姓名）
                for s in (str(f), str(f).replace("\\", "\\\\"), f.name, f.stem):
                    msg = msg.replace(s, "<文件>")
            print(f"\n[!] [{n}/{len(files)}] 失败：{f.name if args.include_identity else f'第 {n} 个文件'}：{msg}")
            continue
        rows.append(summary_row(rep))
        rh, mo = rep["rhythm"], rep["morphology"] or {}
        print(f"    心率 {rh['hr_bpm']}（手表 {rh['reported_bpm']}），心搏 {rh['beats']}，"
              f"提前 同形 {rh['n_premature_same']} / 异形 {rh['n_premature_diff']}，长间歇 {len(rh['pauses'])}，"
              f"QTc {mo.get('qtc_bazett_ms')}/{mo.get('qtc_fridericia_ms')}，RMSSD {rep['hrv']['rmssd_ms']}")
        for fl in rep["flags"]:
            print("    -", fl)
    if len(rows) > 1:
        df = pd.DataFrame(rows).sort_values("记录时间", na_position="last")
        df.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n[OK] 汇总表：{out / 'summary.csv'}")
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    sys.exit(main())
