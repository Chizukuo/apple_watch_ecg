# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Chizukuo
"""
测试全部基于合成数据（tests/synth.py），不含任何真实记录或个人信息。

运行：  pytest -q
本地自检（可选）：把自己的 PDF 放在某个文件夹，设置环境变量后运行，
         只用 PDF 自带的手表结果做对照，不需要也不会保存任何期望值：
         ECG_LOCAL_DIR=/path/to/pdfs pytest -q -k local
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import apple_watch_ecg as A                                       # noqa: E402
from synth import (add_motion_artifact, build_apple_like_pdf, make_beats, make_rsa_beats, synth_signal,  # noqa: E402
                   waves_variant)

PREM = (A.LABEL_PREM_SAME, A.LABEL_PREM_DIFF)
FAKE_NAME = "测试用户"
FAKE_FILE = "健康 - 测试用户 - 心电图2026-10-13 11.25.pdf"     # 仿 Apple 导出文件名（含姓名）


def make_pdf(tmp_path, name=FAKE_FILE, beats=None, hr=72.0, events=None, alt=None, **pdf_kw):
    beats = beats or make_beats(hr=hr, events=events, alt=alt)
    t, v = synth_signal(beats)
    path = tmp_path / name
    pdf_kw.setdefault("bpm", round(60 * (len(beats) - 1) / (beats[-1][0] - beats[0][0])))
    build_apple_like_pdf(path, t, v, **pdf_kw)
    return path, beats


def run(path, exclude=None):
    rec = A.load(path)
    return rec, A.analyze(rec, exclude)


def labels_near(res, t0, tol=0.06):
    return [l for tt, l in zip(res["times"], res["labels"]) if abs(tt - t0) < tol]


# ---------------------------------------------------------------------------
# 提取与基本测量
# ---------------------------------------------------------------------------
def test_sinus_roundtrip(tmp_path):
    path, beats = make_pdf(tmp_path)
    rec, res = run(path)
    rep = res["report"]
    assert rec.source == "vector" and rec.fs == 512 and len(rec.rows) == 3
    assert rec.record_id == "ECG_20261013_1125"
    assert len(res["pos"]) == len(beats)                                   # 不多检、不漏检
    for t0, _ in beats:
        assert min(abs(res["times"] - t0)) < 0.03
    assert set(res["labels"]) == {A.LABEL_NORMAL}
    assert abs(rep["rhythm"]["hr_bpm"] - 72) < 0.6
    mo = rep["morphology"]
    assert abs(mo["r_amp_mv"] - 0.94) < 0.08
    assert abs(mo["t_amp_mv"] - 0.30) < 0.05
    assert mo["p_measurable"] and mo["p_polarity"] == "直立"
    assert 45 <= mo["qrs_ms"] <= 110
    assert 300 <= mo["qt_ms"] <= 460
    assert rep["hrv"]["rmssd_ms"] < 10
    assert not rep["rhythm"]["pauses"] and not rep["rhythm"]["rr_alternation_start_s"]


@pytest.mark.parametrize("hr", [55, 68, 83, 97, 118])
def test_row_boundaries_no_false_or_missed_beats(tmp_path, hr):
    """三行拼接后，10 秒、20 秒行边界附近不应多出 T 波误检，也不应漏检。"""
    path, beats = make_pdf(tmp_path, hr=hr, label="窦性心律" if hr < 100 else "高心率")
    _, res = run(path)
    assert len(res["pos"]) == len(beats)
    rr = np.diff(res["times"])
    assert rr.min() > 0.8 * 60 / hr
    assert not [l for l in res["labels"] if l != A.LABEL_NORMAL]


# ---------------------------------------------------------------------------
# 异常搏动与节律
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["V", "W"])
def test_ventricular_premature_beat_is_reported(tmp_path, kind):
    """宽大畸形的提前搏动（负向、正向）都要被检测、标为"提前·QRS异形"，代偿间歇不能被报成长间歇。"""
    path, beats = make_pdf(tmp_path, events={8: kind})
    _, res = run(path)
    t_pvc = [t for t, k in beats if k == kind][0]
    assert labels_near(res, t_pvc) == [A.LABEL_PREM_DIFF]
    rh = res["report"]["rhythm"]
    assert rh["n_premature_diff"] == 1 and rh["n_premature_same"] == 0
    assert not rh["pauses"]
    assert any("室性早搏" in f for f in res["report"]["flags"])


def test_atrial_premature_beat_is_reported(tmp_path):
    path, beats = make_pdf(tmp_path, events={8: "A"})
    _, res = run(path)
    t_pac = [t for t, k in beats if k == "A"][0]
    assert labels_near(res, t_pac) == [A.LABEL_PREM_SAME]
    assert res["report"]["rhythm"]["n_premature_diff"] == 0


def test_bigeminy_pattern(tmp_path):
    # 连续 4 个窦性心搏后各跟一个早搏：N V N V N V N V（二联律）
    path, _ = make_pdf(tmp_path, events={6: "V", 7: "V", 8: "V", 9: "V"})
    _, res = run(path)
    assert res["report"]["rhythm"]["n_premature_diff"] == 4
    assert res["report"]["rhythm"]["bigeminy_start_s"] is not None
    # 早搏之间隔两个窦性心搏（三联律）不应报二联律
    path3, _ = make_pdf(tmp_path, name="tri.pdf", events={6: "V", 8: "V", 10: "V"})
    _, res3 = run(path3)
    assert res3["report"]["rhythm"]["n_premature_diff"] == 3
    assert res3["report"]["rhythm"]["bigeminy_start_s"] is None


def test_sinus_pause_is_reported(tmp_path):
    path, _ = make_pdf(tmp_path, events={10: "pause"})
    _, res = run(path)
    rh = res["report"]["rhythm"]
    assert len(rh["pauses"]) == 1 and rh["pauses"][0]["ratio"] > 1.9
    assert rh["n_premature_same"] == rh["n_premature_diff"] == 0
    assert any("长间歇" in f for f in res["report"]["flags"])


def test_rr_alternation_is_flagged_not_premature(tmp_path):
    path, _ = make_pdf(tmp_path, alt=(4, 20))
    _, res = run(path)
    rh = res["report"]["rhythm"]
    assert rh["rr_alternation_start_s"]
    assert rh["n_premature_same"] == rh["n_premature_diff"] == 0


@pytest.mark.parametrize("hr", [70, 95, 125])
def test_respiratory_sinus_arrhythmia_is_not_premature(tmp_path, hr):
    """呼吸引起的 RR 渐变（±15%）不能被当成提前搏动。"""
    beats = make_rsa_beats(hr=hr, depth=0.15)
    path, _ = make_pdf(tmp_path, beats=beats, label="窦性心律" if hr < 100 else "高心率", bpm=hr)
    _, res = run(path)
    assert len(res["pos"]) == len(beats)
    assert set(res["labels"]) == {A.LABEL_NORMAL}
    assert not res["report"]["rhythm"]["pauses"]


def test_motion_artifact_is_excluded_not_flagged(tmp_path):
    beats = make_beats(hr=72)
    t, v = synth_signal(beats)
    v = add_motion_artifact(t, v, at=14.0)
    path = tmp_path / "artifact.pdf"
    build_apple_like_pdf(path, t, v, bpm=72)
    _, res = run(path)
    rep = res["report"]
    assert rep["quality"]["artifact_windows_s"]
    a0, a1 = rep["quality"]["artifact_windows_s"][0]
    assert a0 <= 14.0 <= a1
    assert not rep["rhythm"]["pauses"]
    assert all(l in (A.LABEL_NORMAL, A.LABEL_REMOVED) for l in res["labels"])


def test_excluded_beat_is_not_reported_as_pause(tmp_path):
    path, beats = make_pdf(tmp_path)
    t_mid = beats[len(beats) // 2][0]
    _, res = run(path, exclude=[(t_mid - 0.1, t_mid + 0.1)])
    rep = res["report"]
    assert not rep["rhythm"]["pauses"]
    assert not any("长间歇" in f for f in rep["flags"])
    assert any("被排除" in f and "不会被当作停顿" in f for f in rep["flags"])
    assert rep["rhythm"]["n_removed"] == 1


# ---------------------------------------------------------------------------
# 手表信息的利用
# ---------------------------------------------------------------------------
def test_pdf_metadata_is_used(tmp_path):
    path, _ = make_pdf(tmp_path, symptoms="心悸", dob="2006年1月1日", age=20)
    rec, res = run(path)
    w = rec.watch
    assert w["recorded_at"] == "2026-10-13 11:25" and w["age_at_recording"] == 20
    assert w["category"] == "sinus" and w["algorithm_version"] == 2 and w["device_model"] == "Watch8,5"
    assert w["symptoms"] == "心悸" and w["ios_build"] == "24A446" and w["exported_at"]
    flags = res["report"]["flags"]
    assert any("22 岁以下" in f for f in flags)
    assert any("心悸" in f for f in flags)


def test_adult_has_no_age_flag(tmp_path):
    path, _ = make_pdf(tmp_path)
    _, res = run(path)
    assert not any("22 岁以下" in f for f in res["report"]["flags"])


def test_watch_classification_consistency(tmp_path):
    # 信号约 120 次/分，但 PDF 写"窦性心律、72 次/分"：两项检查都应报警
    path, _ = make_pdf(tmp_path, hr=120, label="窦性心律", bpm=72)
    _, res = run(path)
    flags = res["report"]["flags"]
    assert any("相差超过 3 次/分" in f for f in flags)
    assert any("对应 50–99 次/分" in f for f in flags)
    # 正确标注为"高心率"时不报警
    path2, _ = make_pdf(tmp_path, name="b.pdf", hr=120, label="高心率", bpm=120)
    _, res2 = run(path2)
    assert not any("不一致" in f or "相差超过" in f for f in res2["report"]["flags"])


def test_analyze_is_repeatable(tmp_path):
    path, _ = make_pdf(tmp_path, events={8: "V"})
    rec = A.load(path)
    r1, r2 = A.analyze(rec)["report"], A.analyze(rec)["report"]
    assert r1["notes"] == r2["notes"] and r1["rhythm"] == r2["rhythm"] and r1["flags"] == r2["flags"]


def test_watch_category_mapping():
    assert A.watch_category("窦性心律") == "sinus"
    assert A.watch_category("高心率") == "high_hr"
    assert A.watch_category("房颤") == "afib"
    assert A.watch_category("房颤 — 高心率") == "afib_high"
    assert A.watch_category("心率过低或过高") == "low_or_high"
    assert A.watch_category("不确定") == "inconclusive"
    assert A.watch_category(None) == "unknown"


# ---------------------------------------------------------------------------
# 隐私、命令行、编码
# ---------------------------------------------------------------------------
def _all_text(out_dir: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8-sig", errors="ignore")
                     for p in out_dir.iterdir() if p.suffix in (".json", ".csv"))


def test_outputs_have_no_identity_by_default(tmp_path):
    p1, _ = make_pdf(tmp_path)
    p2, _ = make_pdf(tmp_path, name="健康 - 测试用户 - 心电图2026-10-14 00.51.pdf", recorded="2026年10月14日00:51")
    out = tmp_path / "out"
    assert A.main([str(p1), str(p2), "-o", str(out)]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert all(n.startswith("ECG_2026") or n == "summary.csv" for n in names), names
    text = _all_text(out)
    for secret in (FAKE_NAME, "2000-01-01", "2000年1月1日", "心电图2026"):
        assert secret not in text
    rep = json.loads((out / "ECG_20261013_1125_report.json").read_text(encoding="utf-8"))
    assert "identity" not in rep
    assert A.dashboard_title(rep).find(FAKE_NAME) == -1


def test_include_identity_is_opt_in(tmp_path):
    p1, _ = make_pdf(tmp_path)
    out = tmp_path / "out"
    assert A.main([str(p1), "-o", str(out), "--include-identity"]) == 0
    rep = json.loads((out / "ECG_20261013_1125_report.json").read_text(encoding="utf-8"))
    assert rep["identity"]["name"] == FAKE_NAME and rep["identity"]["dob"] == "2000-01-01"
    assert rep["identity"]["source_file"] == FAKE_FILE


def test_duplicate_record_times_get_unique_ids(tmp_path):
    p1, _ = make_pdf(tmp_path, name="a.pdf")
    p2, _ = make_pdf(tmp_path, name="b.pdf")
    out = tmp_path / "out"
    assert A.main([str(p1), str(p2), "-o", str(out)]) == 0
    assert (out / "ECG_20261013_1125_report.json").exists() and (out / "ECG_20261013_1125_2_report.json").exists()


def test_exclude_rejected_in_batch(tmp_path):
    p1, _ = make_pdf(tmp_path, name="a.pdf")
    p2, _ = make_pdf(tmp_path, name="b.pdf")
    assert A.main([str(p1), str(p2), "--exclude", "0-4", "-o", str(tmp_path / "out")]) == 2


def test_bad_exclude_and_empty_input_are_rejected(tmp_path):
    p1, _ = make_pdf(tmp_path, name="a.pdf")
    assert A.main([str(p1), "--exclude", "0-4-5", "-o", str(tmp_path / "out")]) == 2
    (tmp_path / "empty").mkdir()
    assert A.main([str(tmp_path / "empty"), "-o", str(tmp_path / "out")]) == 2


def test_failed_file_name_is_not_printed_by_default(tmp_path, capsys):
    """Apple 导出的文件名里带姓名：失败时也不能打印出来。"""
    bad = tmp_path / FAKE_FILE
    bad.write_bytes(b"not a pdf")
    assert A.main([str(bad), "-o", str(tmp_path / "out")]) == 1
    assert FAKE_NAME not in capsys.readouterr().out


def test_gbk_redirected_output_does_not_crash(tmp_path):
    """中文 Windows 把输出重定向到文件时使用 GBK 编码，旧版会因 ✓ 字符崩溃。"""
    p1, _ = make_pdf(tmp_path, name="a.pdf")
    p2, _ = make_pdf(tmp_path, name="b.pdf", recorded="2026年10月14日00:51")
    env = dict(os.environ, PYTHONIOENCODING="gbk")
    r = subprocess.run([sys.executable, str(ROOT / "apple_watch_ecg.py"), str(p1), str(p2), "-o", str(tmp_path / "out")],
                       env=env, capture_output=True)
    assert r.returncode == 0, r.stderr.decode("gbk", "replace")
    assert (tmp_path / "out" / "summary.csv").exists()


# ---------------------------------------------------------------------------
# 光栅（截图）路径
# ---------------------------------------------------------------------------
def test_raster_path_detects_premature_beat(tmp_path):
    path, beats = make_pdf(tmp_path, events={8: "V"})
    png = tmp_path / "截图.png"
    pymupdf.open(path)[0].get_pixmap(dpi=200).save(png)
    rec, res = run(png)
    assert rec.source == "raster" and rec.record_id.startswith("ECG_")
    t_pvc = [t for t, k in beats if k == "V"][0]
    assert abs(res["report"]["rhythm"]["hr_bpm"] - 60 * (len(beats) - 1) / (beats[-1][0] - beats[0][0])) < 2
    assert labels_near(res, t_pvc, tol=0.08) == [A.LABEL_PREM_DIFF]


# ---------------------------------------------------------------------------
# 本地自检（可选）：只拿 PDF 自带的手表结果做对照
# ---------------------------------------------------------------------------
LOCAL = os.environ.get("ECG_LOCAL_DIR")


# ---------------------------------------------------------------------------
# 形态提醒：先在波形上验证“测得出来”，再用构造的测量值验证阈值逻辑
# ---------------------------------------------------------------------------
def analyze_waves(hr=72.0, sex="unknown", placement="wrist", **kw):
    t, v = synth_signal(make_beats(hr=hr), waves=waves_variant(**kw))
    rec = A.Recording(Path("synthetic"), {}, {}, [], "synthetic", A.DEFAULT_FS, "synthetic", t, v, np.zeros(len(v), bool))
    rep = A.analyze(rec, sex=sex, placement=placement)["report"]
    return rep, [a["code"] for a in rep["morphology_alerts"]["alerts"]]


def test_normal_waveform_has_no_morphology_alert():
    rep, codes = analyze_waves()
    assert codes == []
    assert not any(f.startswith("【") for f in rep["flags"])


@pytest.mark.parametrize("kw,code", [
    (dict(t_shift=0.07), "QTC_LONG"),
    (dict(t_shift=0.12), "QTC_LONG_MARKED"),
    (dict(t_shift=-0.04), "QTC_SHORT"),
    (dict(t_shift=-0.07), "QTC_SHORT_MARKED"),
    (dict(qrs_scale=2.0), "QRS_WIDE"),
    (dict(p_at=-0.27), "PR_LONG"),
    (dict(p_at=-0.10), "PR_SHORT"),
    (dict(st=0.25), "ST_ELEVATION"),
    (dict(st=-0.20), "ST_DEPRESSION"),
    (dict(t_amp=-0.30), "T_INVERSION"),
])
def test_morphology_abnormality_is_detected(kw, code):
    rep, codes = analyze_waves(**kw)
    assert codes == [code], (kw, codes, rep["morphology"])


@pytest.mark.parametrize("hr", [50, 58, 66])
def test_tall_t_wave_is_not_counted_as_a_beat(hr):
    # T 波比 R 波还高时，常规 R 峰检测会把 T 波也当成心搏（MIT-BIH 113 号记录即如此）
    beats = make_beats(hr=hr)
    t, v = synth_signal(beats, waves=waves_variant(t_amp=1.2, t_sd=0.05, t_shift=0.05))
    raw, _ = A._clean_and_detect(v, A.DEFAULT_FS)
    assert len(raw) > len(beats)                            # 确认这个场景确实会多检
    rec = A.Recording(Path("synthetic"), {}, {}, [], "synthetic", A.DEFAULT_FS, "synthetic", t, v, np.zeros(len(v), bool))
    rh = A.analyze(rec)["report"]["rhythm"]
    assert rh["beats"] == len(beats)
    assert rh["n_premature_same"] + rh["n_premature_diff"] == 0 and rh["pauses"] == []


def test_slow_terminal_s_wave_counts_as_wide_qrs():
    # 右束支阻滞样：R 波后紧跟一个宽而缓的 S 波。基本的斜率规则会把 J 点停在 S 波之前
    w = waves_variant()
    w["N"] = w["N"] + [("S2", -0.5, 0.065, 0.022)]
    t, v = synth_signal(make_beats(), waves=w)
    rec = A.Recording(Path("synthetic"), {}, {}, [], "synthetic", A.DEFAULT_FS, "synthetic", t, v, np.zeros(len(v), bool))
    rep = A.analyze(rec)["report"]
    assert rep["morphology"]["qrs_ms"] >= 120
    assert [a["code"] for a in rep["morphology_alerts"]["alerts"]] == ["QRS_WIDE"]
    assert rep["rhythm"]["n_odd"] == 0                      # 每拍都一样宽，不应被标为“形态异常”


def test_flat_t_marks_qt_unreliable():
    rep, codes = analyze_waves(t_amp=0.04, t_shift=0.12)
    assert codes == [] and rep["morphology"]["qt_reliable"] is False


def test_marked_alert_is_listed_first():
    rep, codes = analyze_waves(t_shift=0.12)
    assert rep["flags"][0].startswith("【建议就医】QTc 明显延长")


def test_qtc_threshold_depends_on_sex():
    _, male = analyze_waves(t_shift=0.05, sex="male")          # QTcB 约 453
    _, female = analyze_waves(t_shift=0.05, sex="female")
    _, unknown = analyze_waves(t_shift=0.05)
    assert male == ["QTC_LONG"] and unknown == ["QTC_LONG"] and female == []


def test_non_wrist_placement_skips_st_and_t():
    rep, codes = analyze_waves(st=0.25, placement="other")
    assert codes == []
    assert any("不在手腕上" in s for s in rep["morphology_alerts"]["skipped"])


def test_wide_qrs_suppresses_secondary_st_t_alerts():
    _, codes = analyze_waves(qrs_scale=2.6, t_amp=-0.3)
    assert "QRS_WIDE" in codes and "T_INVERSION" not in codes and "ST_ELEVATION" not in codes


def test_fast_rate_does_not_create_st_t_or_pr_alerts():
    # 心率快时 TP 段混进上一拍的 T 波：ST、T、PR 不能因此误报
    for hr in (110, 130):
        rep, codes = analyze_waves(hr=hr)
        assert not {"ST_ELEVATION", "ST_DEPRESSION", "T_INVERSION", "PR_SHORT", "PR_LONG"} & set(codes), (hr, codes)


def _mo(**kw):
    base = dict(template_beats=30, qtc_bazett_ms=420, qtc_fridericia_ms=410, qt_beat_iqr_ms=[380, 395], t_amp_mv=0.3,
                t_polarity="直立", qrs_ms=90, p_measurable=True, pr_ms=160, st_j_mv=0.0, st_j60_mv=0.05,
                r_amp_mv=1.0, s_amp_mv=-0.3)
    base.update(kw)
    return base


@pytest.mark.parametrize("qtc,sex,expect", [
    (449, "male", []), (450, "male", ["QTC_LONG"]), (459, "female", []), (460, "female", ["QTC_LONG"]),
    (450, "unknown", ["QTC_LONG"]), (499, "male", ["QTC_LONG"]), (500, "female", ["QTC_LONG_MARKED"]),
    (361, "male", []), (360, "male", ["QTC_SHORT"]), (341, "male", ["QTC_SHORT"]), (340, "male", ["QTC_SHORT_MARKED"]),
])
def test_qtc_thresholds(qtc, sex, expect):
    al, _ = A.morphology_alerts(_mo(qtc_bazett_ms=qtc), hr=70, sex=sex)
    assert [a["code"] for a in al] == expect


def test_fridericia_is_used_above_100_bpm():
    mo = _mo(qtc_bazett_ms=480, qtc_fridericia_ms=430)        # 心率快时 Bazett 偏高
    assert A.morphology_alerts(mo, hr=99)[0][0]["code"] == "QTC_LONG"
    assert A.morphology_alerts(mo, hr=115)[0] == []


@pytest.mark.parametrize("kw,reason", [
    (dict(qtc_bazett_ms=520, t_amp_mv=0.05), "T 波太低"),
    (dict(qtc_bazett_ms=520, qt_beat_iqr_ms=[360, 420]), "波动太大"),
    (dict(qtc_bazett_ms=None), "未能测出"),
])
def test_unreliable_qt_is_skipped_not_alerted(kw, reason):
    al, skipped = A.morphology_alerts(_mo(**kw), hr=70)
    assert al == [] and any(reason in s for s in skipped)


def test_pr_rules_need_measurable_p_and_normal_rate():
    assert A.morphology_alerts(_mo(pr_ms=230, p_measurable=False), hr=70)[0] == []
    assert A.morphology_alerts(_mo(pr_ms=230), hr=110)[0] == []
    assert [a["code"] for a in A.morphology_alerts(_mo(pr_ms=230), hr=70)[0]] == ["PR_LONG"]


def test_st_depression_needs_horizontal_or_downsloping():
    up = _mo(st_j_mv=-0.15, st_j60_mv=0.02)                     # J 点压低、快速上斜：多为心率快的正常表现
    flat = _mo(st_j_mv=-0.15, st_j60_mv=-0.12)
    assert A.morphology_alerts(up, hr=70)[0] == []
    assert [a["code"] for a in A.morphology_alerts(flat, hr=70)[0]] == ["ST_DEPRESSION"]


def test_t_inversion_only_with_upright_qrs():
    neg_qrs = _mo(t_polarity="倒置", t_amp_mv=-0.3, r_amp_mv=0.3, s_amp_mv=-1.0)
    assert A.morphology_alerts(neg_qrs, hr=70)[0] == []


def test_few_template_beats_are_marked_low_confidence():
    al, _ = A.morphology_alerts(_mo(qrs_ms=130, template_beats=3), hr=70)
    assert "可信度低" in al[0]["text"]


def test_invalid_sex_or_placement_rejected():
    with pytest.raises(ValueError):
        A.morphology_alerts(_mo(), hr=70, sex="M")
    with pytest.raises(ValueError):
        A.morphology_alerts(_mo(), hr=70, placement="chest")


def test_cli_passes_sex_and_placement(tmp_path):
    path, _ = make_pdf(tmp_path)
    out = tmp_path / "out"
    assert A.main([str(path), "-o", str(out), "--sex", "female", "--placement", "other"]) == 0
    rep = json.loads(next(out.glob("*_report.json")).read_text(encoding="utf-8"))
    assert rep["morphology_alerts"]["sex"] == "female" and rep["morphology_alerts"]["placement"] == "other"
    assert any("不在手腕上" in f for f in rep["flags"])



@pytest.mark.skipif(not LOCAL, reason="设置 ECG_LOCAL_DIR 指向本地 PDF 文件夹后运行")
def test_local_pdfs_agree_with_watch():
    pdfs = sorted(Path(LOCAL).glob("*.pdf"))
    assert pdfs, "文件夹里没有 PDF"
    for p in pdfs:
        rec, res = run(p)
        rep = res["report"]
        bpm = rec.watch.get("reported_bpm")
        if bpm:
            assert abs(rep["rhythm"]["hr_bpm"] - bpm) <= 3, rec.record_id
        assert not any("不一致" in f for f in rep["flags"]), rec.record_id
