# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Chizukuo
"""在自己的真实记录里注入已知异常，检查工具能否在自己的波形和噪声上检出。

用法：python validation/inject_real.py 你的PDF文件夹

对每份记录，在 3 个位置（开头、约 10 秒的行边界、约 21 秒）分别注入：
  V     室性早搏：0.6 RR 处叠加宽大、负向的 QRS，并抹掉下一拍窦性心搏（代偿间歇）
  A     房性早搏：剪掉第 k 拍之前 0.25 RR 的 TP 段（提前 25%，窦房结重整）
  pause 窦性停搏：在第 k 拍之前插入一个 RR 长的平直基线
统计检出数和“别处的多余标注”。只在屏幕上输出记录编号，不输出身份信息。
"""
import sys, warnings, glob; warnings.filterwarnings('ignore')
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
import numpy as np, apple_watch_ecg as A
from dataclasses import replace
fs=512
def g(t,a,mu,sd): return a*np.exp(-0.5*((t-mu)/sd)**2)
def rebuild(rec, v): 
    return replace(rec, t=rec.t[0]+np.arange(len(v))/fs, v=v, clip=np.zeros(len(v),bool), notes=list(rec.notes))
if len(sys.argv) != 2: sys.exit(__doc__)
rows=[]
for f in sorted(glob.glob(str(_P(sys.argv[1]) / '*.pdf'))):
    rec=A.load(f); base=A.analyze(rec); times=base['times']; labels=base['labels']
    rr=np.median(np.diff(times)); amp=np.percentile(np.abs(rec.v),99.5)
    removed=base['removed']
    ok=[k for k in range(2,len(times)-3) if labels[k-1]=='正常' and labels[k]=='正常' and labels[k+1]=='正常'
        and not any(a-1<=times[k]<=b+1 for a,b in removed)]
    picks=[ok[1], min(ok,key=lambda k:abs(times[k]-10.0)), min(ok,key=lambda k:abs(times[k]-21.0))]
    for k in picks:
        tk=times[k]; v=rec.v.copy(); t0=rec.t[0]; i=int(round((tk-t0)*fs)); n=int(round(rr*fs))
        # 1) 室性早搏：0.6 RR 处叠加宽 QRS（负向、幅度与本人 R 波相当），下一拍窦性心搏抹平 → 代偿间歇
        vp=v.copy(); j=i+int(0.6*n); nx=int(round((times[k+1]-t0)*fs))
        tt=np.arange(len(v))/fs  # 相对索引
        seg=slice(nx-int(0.25*fs), nx+int(0.40*fs))
        vp[seg]=np.linspace(vp[seg.start], vp[seg.stop], seg.stop-seg.start)   # 去掉下一拍
        vp+= g(tt,0.4*amp,(j/fs)-0.04,0.025)+g(tt,-1.4*amp,j/fs+0.02,0.03)+g(tt,0.45*amp,j/fs+0.3,0.06)
        # 2) 房性早搏：把第 k 拍之前的 TP 段剪掉 0.25 RR（窦房结重整，非代偿）
        cut=int(0.25*n); c0=i-int(0.18*fs)-cut
        va=np.concatenate([v[:c0], v[c0+cut:]]); va=np.concatenate([va, np.full(cut, va[-1])])
        # 3) 窦性停搏：在第 k 拍前插入 1 个 RR 的平直基线
        c1=i-int(0.18*fs); vs=np.concatenate([v[:c1], np.full(n, v[c1]), v[c1:]])[:len(v)]
        for kind, vv, ttrue in [('V',vp,t0+j/fs),('A',va,t0+(i-cut)/fs),('pause',vs,None)]:
            try: res=A.analyze(rebuild(rec,vv),sex='male')
            except Exception as e: rows.append((rec.record_id,kind,round(tk,1),'ERR '+str(e))); continue
            rh=res['report']['rhythm']; L=res['labels']; T=res['times']
            if kind=='pause':
                hit=bool(rh['pauses']); extra=rh['n_premature_same']+rh['n_premature_diff']+rh['n_odd']
            else:
                near=[l for t_,l in zip(T,L) if abs(t_-ttrue)<0.08]
                want=('提前·QRS异形','形态异常') if kind=='V' else ('提前·QRS同形','提前·QRS异形')
                hit=any(l in want for l in near)
                extra=sum(1 for t_,l in zip(T,L) if l not in ('正常','排除段内') and abs(t_-ttrue)>=0.08)+len(rh['pauses'])
            rows.append((rec.record_id,kind,round(tk,1),hit,extra,[a['code'] for a in res['report']['morphology_alerts']['alerts']]))
for r in rows: print(r[0], r[1], f'{float(r[2]):.1f}s', *r[3:])
from collections import defaultdict
agg=defaultdict(lambda:[0,0,0])
for r in rows:
    if isinstance(r[3],bool): agg[r[1]][0]+=r[3]; agg[r[1]][1]+=1; agg[r[1]][2]+=(r[4]>0)
print({k:f'检出 {a}/{b}，有多余标注 {c}/{b}' for k,(a,b,c) in agg.items()})
