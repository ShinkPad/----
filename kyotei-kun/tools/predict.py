#!/usr/bin/env python3
"""ボート基盤モデルで1レースを予想する（official.py のデータ → 各艇の1着確率 → 3連単・2連単の確率）

使い方: python3 predict.py <場番号> <R> <YYYYMMDD> [--top N]
重みは ../data/model_weights.json（model.py --save で作る）
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import model as M
import official as O
import stats as S

PTS = {"1": 10, "2": 8, "3": 6, "4": 4, "5": 2, "6": 1}


def num(s, default=0.0):
    m = re.search(r"-?\d+(\.\d+)?", str(s))
    return float(m.group()) if m else default


def features(d):
    boats = d["出走表"]["艇"]
    ex = {b["枠"]: num(b["展示"], None) if b["展示"] else None for b in d["直前情報"]["艇"]}
    if any(v is None for v in ex.values()):
        ex = {k: 6.8 for k in ex}  # 展示前は全艇同じ扱い
    mean_ex = sum(ex.values()) / 6
    ex_rank = {b: i + 1 for i, b in enumerate(sorted(ex, key=ex.get))}
    nat = {b["枠"]: num(b["全国勝率"]) for b in boats}
    nat_mean = sum(nat.values()) / 6
    mot = {b["枠"]: num(b["モーター2連率"]) for b in boats}
    mot_top = max(mot, key=mot.get)
    title = " ".join(d["出走表"]["タイトル"])
    cat = S.race_category(title)
    w = d["直前情報"]["気象"]
    sm = d.get("当日傾向", {})
    n_before = sm.get("レース数", 0)
    by = sm.get("コース別着数", {})
    feats = {}
    for b in boats:
        k = b["枠"]
        age, wt = (num(x) for x in b["年齢/体重"].split("/"))
        hist = [PTS[x["着"]] for x in b["今節"] if x["着"] in PTS]
        loc = num(b["当地勝率"])
        trend_w = (by.get(k, by.get(str(k), [0]))[0] / n_before) if n_before else 0.0
        feats[k] = {
            "course": k, "nat": nat[k] - nat_mean, "loc": (loc - nat[k]) if loc > 0 else 0.0,
            "loc_none": 1.0 if loc == 0 else 0.0, "nat2": (num(b["全国2連率"]) - 35) / 10,
            "cls": {"A1": 0, "A2": 1, "B1": 2, "B2": 3}.get(b["級別"], 2),
            "mot": (mot[k] - 35) / 10, "mot_top": 1.0 if k == mot_top else 0.0,
            "boat": (num(b["ボート2連率"], 35) - 35) / 10,
            "ex": (mean_ex - ex[k]) * 10, "ex_top": 1.0 if ex_rank[k] == 1 else 0.0,
            "weight": (wt - 52) / 3, "age": (age - 40) / 10,
            "form": (sum(hist) / len(hist) - 5.5) / 3 if hist else 0.0, "form_n": min(len(hist), 8) / 8,
            "wind": num(w.get("風速")), "wave": num(w.get("波高")), "trend": trend_w,
            "final": 1.0 if cat == "優勝戦" else 0.0,
            "seed": 1.0 if cat in ("優勝戦", "ドリーム戦", "特選・選抜", "その他（場独自の企画名）") else 0.0,
            "yosen": 1.0 if cat == "予選" else 0.0,
        }
    return feats, cat


def predict(jcd, rno, hd, top=12):
    here = os.path.dirname(os.path.abspath(__file__))
    W = json.load(open(os.path.join(here, "..", "data", "model_weights.json"), encoding="utf-8"))
    d = O.collect(jcd, rno, hd, 200)
    feats, cat = features(d)
    X = M.matrix([{"feats": feats}], W["groups"])
    p = M.probs(X, (np.array(W["weights"]), np.array(W["mean"]), np.array(W["std"])))[0]
    t3 = M.harville(p)
    t2 = defaultdict(float)
    for k, v in t3.items():
        t2[k[:3]] += v
    odds3 = {k: float(v) for k, v in d["3連単オッズ"].items() if re.fullmatch(r"[\d.]+", v)}
    odds2 = {k: float(v) for k, v in d.get("2連単オッズ", {}).items() if re.fullmatch(r"[\d.]+", v)}
    names = {b["枠"]: b["選手"] for b in d["出走表"]["艇"]}
    out = [f"# {d['場']} {rno}R ボート基盤モデル（{cat}）", "", "| 枠 | 選手 | 1着確率 | 3着以内確率 |", "|---|---|---|---|"]
    top3 = defaultdict(float)
    for k, v in t3.items():
        for b in k.split("-"):
            top3[int(b)] += v
    for i in range(6):
        out.append(f"| {i + 1} | {names[i + 1]} | {100 * p[i]:.1f}% | {100 * top3[i + 1]:.1f}% |")
    out += ["", "## 2連単（確率順）", "| 買い目 | 確率 | オッズ | 期待値 |", "|---|---|---|---|"]
    for k in sorted(t2, key=t2.get, reverse=True)[:8]:
        o = odds2.get(k)
        out.append(f"| {k} | {100 * t2[k]:.1f}% | {o or '-'} | {t2[k] * o:.2f} |" if o else f"| {k} | {100 * t2[k]:.1f}% | - | - |")
    out += ["", "## 3連単（確率順）", "| 買い目 | 確率 | オッズ | 期待値 |", "|---|---|---|---|"]
    for k in sorted(t3, key=t3.get, reverse=True)[:top]:
        o = odds3.get(k)
        out.append(f"| {k} | {100 * t3[k]:.1f}% | {o or '-'} | {t3[k] * o:.2f} |" if o else f"| {k} | {100 * t3[k]:.1f}% | - | - |")
    return "\n".join(out), p, t3


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 12
    print(predict(int(a[0]), int(a[1]), a[2], top)[0])
