"""用 MIT-BIH 心律失常数据库做外部验证。

数据库本身不随仓库分发：从 PhysioNet 下载 mitdb（https://physionet.org/content/mitdb/1.0.0/），
解压后把目录传给 --data。

做法
----
* 取 MLII 导联，切成互不重叠的 30 秒窗口（与 Apple Watch 一次记录同长），360 Hz 重采样到 512 Hz，
  直接送进 apple_watch_ecg.analyze()（跳过 PDF 提取这一步，验证的是检测与判读逻辑）。
* 起搏记录 102、104、107、217 按 AAMI EC57 惯例排除。
* 心搏按 AAMI 分类：N（N L R e j）、S（A a J S）、V（V E）、F、Q。
* 工具检出的心搏与专家标注在 ±150 ms 内配对。

报告
----
* 心搏检出：灵敏度 / 阳性预测值。
* 逐拍：V 被标为“异常”（提前·异形、提前·同形、形态异常任一）的比例；S 被标为“提前”的比例；
  N 被误标为异常的比例（按房颤窗口与非房颤窗口分开）。
* 逐窗口（工具实际给用户看的东西）：
  - 含 V 的窗口里，有“提前/形态异常”提示的比例；含 S 的窗口（非房颤）同理；
  - 纯正常窗口（只有 N 类心搏、窦性心律、没有噪声标注）里，出现任何节律提示的比例 = 误报率。

用法
----
    python validation/mitdb_validate.py --data path/to/mitdb --out results.json [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apple_watch_ecg as awe  # noqa: E402

FS = awe.DEFAULT_FS
WIN_S = 30
PACED = {"102", "104", "107", "217"}
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"}, "F": "F", "/": "Q",
        "f": "Q", "Q": "Q"}
TOL = 0.15
ABNORMAL = {awe.LABEL_PREM_SAME, awe.LABEL_PREM_DIFF, awe.LABEL_ODD}
PREM = {awe.LABEL_PREM_SAME, awe.LABEL_PREM_DIFF}


def records(data: Path) -> list[str]:
    return sorted(p.stem for p in data.glob("*.hea") if p.stem not in PACED)


def run_record(args) -> dict:
    import wfdb
    data, rid = args
    warnings.filterwarnings("ignore")
    rec = wfdb.rdrecord(str(data / rid))
    ann = wfdb.rdann(str(data / rid), "atr")
    ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    sig = rec.p_signal[:, ch]
    fs0 = rec.fs
    sym = np.array(ann.symbol)
    at = ann.sample / fs0
    aux = list(ann.aux_note)

    # 每个时刻的节律（来自 '+' 标注的 aux_note）
    rh_t, rh_v = [0.0], ["(N"]
    for s, x, a in zip(sym, at, aux):
        if s == "+" and a:
            rh_t.append(x); rh_v.append(a.strip("\x00").strip())
    noise_t = at[sym == "~"]

    out = []
    n_win = int(len(sig) / fs0 // WIN_S)
    for w in range(n_win):
        a, b = w * WIN_S, (w + 1) * WIN_S
        seg = sig[int(a * fs0):int(b * fs0)]
        v = resample_poly(seg, 64, 45)          # 360 → 512 Hz
        v = v - np.median(v)
        t = np.arange(len(v)) / FS
        r = awe.Recording(path=Path(f"{rid}_{w}"), watch={}, identity={}, rows=[], source="mitdb", fs=FS,
                          record_id=f"{rid}_{w}", t=t, v=v, clip=np.zeros(len(v), bool))
        # 窗口内的标注
        m = (at >= a) & (at < b)
        beats = [(x - a, AAMI[s]) for s, x in zip(sym[m], at[m]) if s in AAMI]
        idx = np.searchsorted(rh_t, a, side="right") - 1
        rhythms = {rh_v[idx]} | {v_ for t_, v_ in zip(rh_t, rh_v) if a <= t_ < b}
        has_noise = bool(((noise_t >= a) & (noise_t < b)).any())
        item = dict(rec=rid, win=w, rhythms=sorted(rhythms), noise=has_noise,
                    truth=Counter(c for _, c in beats),
                    truth_raw=Counter(s for s in sym[m] if s in AAMI))
        try:
            res = awe.analyze(r)
        except awe.AnalysisError as e:
            item.update(error=str(e))
            out.append(item)
            continue
        rep = res["report"]
        rt = rep["rhythm"]
        det_t = np.asarray(res["times"], float)
        labels = res["labels"]
        # 配对（贪心，按时间）
        used = set()
        pairs = []                              # (真值类别, 工具标签 或 None)
        for x, c in beats:
            if det_t.size == 0:
                pairs.append((c, None)); continue
            j = int(np.argmin(np.abs(det_t - x)))
            if abs(det_t[j] - x) <= TOL and j not in used:
                used.add(j); pairs.append((c, labels[j]))
            else:
                pairs.append((c, None))
        fp_labels = [labels[j] for j in range(len(det_t)) if j not in used]
        item.update(
            pairs=pairs, extra=len(fp_labels), extra_abn=sum(l in ABNORMAL for l in fp_labels),
            flag_prem=rt["n_premature_same"] + rt["n_premature_diff"] > 0,
            flag_prem_diff=rt["n_premature_diff"] > 0,
            flag_odd=rt["n_odd"] > 0, flag_pause=bool(rt["pauses"]),
            flag_alt=bool(rt["rr_alternation_start_s"]), flag_big=rt["bigeminy_start_s"] is not None,
            artifacts=len(rep["quality"]["artifact_windows_s"]),
            hr=rt["hr_bpm"],
            hr_truth=None if len(beats) < 3 else 60 / float(np.mean(np.diff([x for x, _ in beats]))),
            qrs=(rep["morphology"] or {}).get("qrs_ms"),
            qtc=(rep["morphology"] or {}).get("qtc_bazett_ms"),
            alerts=[x["code"] for x in rep["morphology_alerts"]["alerts"]],
        )
        out.append(item)
    return dict(rec=rid, windows=out)


def summarize(results: list[dict]) -> dict:
    wins = [w for r in results for w in r["windows"]]
    ok = [w for w in wins if "error" not in w]
    af = lambda w: any(x in ("(AFIB", "(AFL") for x in w["rhythms"])
    s = {"records": len(results), "windows": len(wins), "windows_failed": len(wins) - len(ok)}

    # 心搏检出
    tp = sum(1 for w in ok for c, l in w["pairs"] if l is not None)
    fn = sum(1 for w in ok for c, l in w["pairs"] if l is None)
    fp = sum(w["extra"] for w in ok)
    s["beat_detection"] = dict(se=round(tp / (tp + fn) * 100, 2), ppv=round(tp / (tp + fp) * 100, 2),
                               tp=tp, fn=fn, fp=fp)

    # 逐拍
    def rate(sel, pred):
        xs = [l for w in ok if sel(w) for c, l in w["pairs"] if pred[0](c) and l is not None]
        hit = sum(pred[1](l) for l in xs)
        return dict(n=len(xs), hit=hit, pct=round(hit / len(xs) * 100, 2) if xs else None)

    s["beat_V_flagged_abnormal"] = rate(lambda w: True, (lambda c: c == "V", lambda l: l in ABNORMAL))
    s["beat_V_flagged_prem_diff_or_odd"] = rate(lambda w: True,
                                                (lambda c: c == "V", lambda l: l in (awe.LABEL_PREM_DIFF, awe.LABEL_ODD)))
    s["beat_S_flagged_premature_nonAF"] = rate(lambda w: not af(w), (lambda c: c == "S", lambda l: l in PREM))
    s["beat_N_flagged_abnormal_nonAF"] = rate(lambda w: not af(w), (lambda c: c == "N", lambda l: l in ABNORMAL))
    s["beat_N_flagged_abnormal_AF"] = rate(af, (lambda c: c == "N", lambda l: l in ABNORMAL))
    # 阳性预测值：工具标“异常”的心搏中，真值确实非 N 的比例
    abn = [(c, l) for w in ok for c, l in w["pairs"] if l in ABNORMAL]
    abn_nonaf = [(c, l) for w in ok if not af(w) for c, l in w["pairs"] if l in ABNORMAL]
    s["beat_abnormal_ppv"] = round(sum(c != "N" for c, _ in abn) / len(abn) * 100, 2) if abn else None
    s["beat_abnormal_ppv_nonAF"] = round(sum(c != "N" for c, _ in abn_nonaf) / len(abn_nonaf) * 100, 2) if abn_nonaf else None

    # 逐窗口
    any_rhythm = lambda w: w["flag_prem"] or w["flag_odd"] or w["flag_pause"] or w["flag_big"]
    def wrate(sel, pred):
        xs = [w for w in ok if sel(w)]
        hit = sum(bool(pred(w)) for w in xs)
        return dict(n=len(xs), hit=hit, pct=round(hit / len(xs) * 100, 2) if xs else None)

    s["win_with_V_alerted"] = wrate(lambda w: w["truth"].get("V", 0) > 0, lambda w: w["flag_prem"] or w["flag_odd"])
    s["win_with_S_nonAF_alerted"] = wrate(lambda w: w["truth"].get("S", 0) > 0 and not af(w), lambda w: w["flag_prem"])
    pure = lambda w: (set(w["truth"]) <= {"N"} and w["rhythms"] == ["(N"] and not w["noise"])
    s["win_pure_normal_any_alert"] = wrate(pure, any_rhythm)
    s["win_pure_normal_premature"] = wrate(pure, lambda w: w["flag_prem"])
    s["win_pure_normal_odd"] = wrate(pure, lambda w: w["flag_odd"])
    s["win_pure_normal_pause"] = wrate(pure, lambda w: w["flag_pause"])
    s["win_pure_normal_alternation"] = wrate(pure, lambda w: w["flag_alt"])
    s["win_pure_normal_with_artifact"] = wrate(pure, lambda w: w["artifacts"] > 0)
    hr_err = [abs(w["hr"] - w["hr_truth"]) for w in ok if pure(w) and w["hr_truth"]]
    s["hr_abs_err_pure_normal_bpm"] = dict(median=round(float(np.median(hr_err)), 2),
                                            p95=round(float(np.percentile(hr_err, 95)), 2), max=round(float(max(hr_err)), 2))
    # 逐记录误报（纯正常窗口）
    per = defaultdict(lambda: [0, 0])
    for w in ok:
        if pure(w):
            per[w["rec"]][0] += 1
            per[w["rec"]][1] += bool(any_rhythm(w))
    s["pure_normal_false_alert_by_record"] = {k: v for k, v in sorted(per.items()) if v[1]}
    # 形态提醒
    # QRS 增宽：以束支传导阻滞窗口（≥90% 的心搏标为 L 或 R）为阳性，以 ≥90% 标为 N 的窗口为阴性
    def frac(w, keys):
        n = sum(w["truth_raw"].values())
        return n and sum(w["truth_raw"].get(k, 0) for k in keys) / n
    bbb = lambda w: frac(w, "LR") >= 0.9 and not af(w)
    nrm = lambda w: frac(w, "N") >= 0.9 and not af(w)
    wide = lambda w: "QRS_WIDE" in w["alerts"]
    s["qrs_wide_alert_in_BBB_windows"] = wrate(bbb, wide)
    s["qrs_wide_alert_in_LBBB_windows"] = wrate(lambda w: frac(w, "L") >= 0.9 and not af(w), wide)
    s["qrs_wide_alert_in_RBBB_windows"] = wrate(lambda w: frac(w, "R") >= 0.9 and not af(w), wide)
    s["qrs_wide_alert_in_normal_conduction_windows"] = wrate(nrm, wide)
    q = defaultdict(list)
    for w in ok:
        if w["qrs"] is not None:
            q["BBB" if bbb(w) else "N" if nrm(w) else "other"].append(w["qrs"])
    s["template_qrs_ms_p25_p50_p75"] = {k: [round(float(np.percentile(v, p)), 0) for p in (25, 50, 75)]
                                        for k, v in q.items()}
    # 各类提醒在“纯正常”窗口里的出现率（MLII 不是导联 I，ST/T 结果仅供参考）
    pn = lambda w: pure(w) and frac(w, "N") == 1          # 只有标为 N 的心搏（排除束支阻滞记录）
    n_pure = sum(1 for w in ok if pn(w))
    alerts = Counter(a for w in ok if pn(w) for a in w.get("alerts", []))
    s["morphology_alerts_in_pure_normal_windows"] = {k: f"{v}/{n_pure}" for k, v in alerts.items()}
    s["morphology_alerts_all_windows"] = dict(Counter(a for w in ok for a in w.get("alerts", [])))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("mitdb_results.json"))
    ap.add_argument("--records", nargs="*")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resummarize", action="store_true", help="只用 --out 里已有的逐窗口结果重算汇总")
    a = ap.parse_args(argv)
    if a.resummarize:
        results = json.loads(a.out.read_text(encoding="utf-8"))["results"]
        for r in results:
            for w in r["windows"]:
                w["truth"], w["truth_raw"] = Counter(w["truth"]), Counter(w.get("truth_raw", {}))
        summ = summarize(results)
        a.out.write_text(json.dumps(dict(summary=summ, results=results), ensure_ascii=False, indent=1, default=str),
                         encoding="utf-8")
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        return 0
    rids = a.records or records(a.data)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_record, [(a.data, r) for r in rids]))
    summ = summarize(results)
    a.out.write_text(json.dumps(dict(summary=summ, results=results), ensure_ascii=False, indent=1, default=str),
                     encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
