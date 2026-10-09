#!/usr/bin/env python3
"""ボート基盤モデル v2 で1レースを予想する。

1. 学習時点（2026-09-30）までの選手・モーターの履歴を読み込み、昨日までの公式データで履歴を最新にする
2. official.py で当日の出走表・直前情報・オッズを取得し、v2 の特徴量を作る
3. 各艇の1着確率 → 2・3着モデル（model3.py）で3連単・2連単の確率 → 期待値で「買い／見送り」を判定（predict.py と同じ基準）

使い方: python3 predict2.py <場番号> <R> <YYYYMMDD> [--budget 円] [--cache キャッシュ]
"""
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import lightgbm as lgb
import numpy as np

import model2 as M2
import model3 as M3
import official as O
import predict as P
import stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DEFAULT_CACHE = os.environ.get("KYOTEI_CACHE", os.path.join(HERE, "..", "..", ".kyotei-cache"))


def num(s, default=np.nan):
    m = re.search(r"-?\d+(\.\d+)?", str(s))
    return float(m.group()) if m else default


def load_history(hd, cache):
    """学習時点の履歴に、学習期間の翌日〜昨日の公式データを足す（結果はキャッシュ）"""
    os.makedirs(cache, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, "model2.txt.json")))
    until = datetime.strptime(meta["history_until"], "%Y-%m-%d").date()
    target = datetime.strptime(hd, "%Y%m%d").date() - timedelta(1)
    snap = os.path.join(cache, f"hist_{target:%Y%m%d}.pkl")
    if os.path.exists(snap):
        return pickle.load(open(snap, "rb"))
    import gzip

    class _U(pickle.Unpickler):
        def find_class(self, module, name):  # model2.py を直接実行して保存した履歴も読めるようにする
            return getattr(M2, name) if name == "History" else super().find_class(module, name)

    hist = _U(gzip.open(os.path.join(DATA, "model2_hist_base.pkl.gz"), "rb")).load()
    if target > until:
        M2.build(until + timedelta(1), target, cache, hist)
    pickle.dump(hist, open(snap, "wb"))
    return hist


def features(d, hist, hd):
    day = datetime.strptime(hd, "%Y%m%d").date()
    boats = {b["枠"]: b for b in d["出走表"]["艇"]}
    bf = d["直前情報"]
    ex = {b["枠"]: num(b["展示"]) for b in bf["艇"]}
    if any(np.isnan(v) or not 6.0 <= v <= 8.0 for v in ex.values()):  # 展示前は列がずれてチルトが入ることがある
        ex = {k: 6.8 for k in boats}
    order = [s["枠"] for s in bf.get("スタート展示", [])]
    course = {b: (order.index(b) + 1 if b in order else b) for b in boats} if len(order) == 6 else {b: b for b in boats}
    by_course = {c: b for b, c in course.items()}
    mean_ex = sum(ex.values()) / 6
    ex_rank = {b: i + 1 for i, b in enumerate(sorted(ex, key=ex.get))}
    nat = {b: num(boats[b]["全国勝率"]) for b in boats}
    nat_rank = {b: i + 1 for i, b in enumerate(sorted(nat, key=nat.get, reverse=True))}
    cat = S.race_category(" ".join(d["出走表"]["タイトル"]))
    w = bf["気象"]
    code = int(num(w.get("風向コード"), 17))
    wind_dir = code if 1 <= code <= 16 else 0
    jcd = S.PLACES.index(d["場"]) + 1
    out = {}
    for b in range(1, 7):
        x = boats[b]
        c = course[b]
        inner = by_course.get(c - 1)
        age, wt = (num(v) for v in x["年齢/体重"].split("/"))
        loc = num(x["当地勝率"], 0)
        f = {
            "venue": jcd, "course": c, "waku": b, "cat": M2.CAT_ORDER.index(cat) if cat in M2.CAT_ORDER else 3,
            "wind_dir": wind_dir, "wind": num(w.get("風速"), 0), "wave": num(w.get("波高"), 0),
            "nat": nat[b], "nat2": num(x["全国2連率"]), "loc": loc if loc > 0 else np.nan, "loc_none": 1 if loc == 0 else 0,
            "cls": {"A1": 0, "A2": 1, "B1": 2, "B2": 3}.get(x["級別"], 2), "age": age, "weight": round(wt),
            "mot2": num(x["モーター2連率"]), "boat2": num(x["ボート2連率"]),
            "ex_diff": mean_ex - ex[b], "ex_rank": ex_rank[b], "nat_rank": nat_rank[b],
            "nat_gap_top": nat[b] - max(v for k, v in nat.items() if k != b),
            "nat_inner": nat[inner] if inner else np.nan, "ex_inner": (mean_ex - ex[inner]) if inner else np.nan,
            "nat_b1": nat[by_course.get(1, 1)], "ex_b1": mean_ex - ex[by_course.get(1, 1)],
        }
        f.update(hist.feats(day, x["登番"], c, jcd, int(num(x["モーター"], 0))))
        out[b] = f
    return out, cat


def predict(jcd, rno, hd, budget=500, cache=DEFAULT_CACHE, log=True):
    meta = json.load(open(os.path.join(DATA, "model2.txt.json")))
    booster = lgb.Booster(model_file=os.path.join(DATA, "model2.txt"))
    hist = load_history(hd, cache)
    d = O.collect(jcd, rno, hd, 200, light=True)
    feats, cat = features(d, hist, hd)
    X = np.array([[feats[b][k] for k in M2.FEATURES] for b in range(1, 7)], dtype=float)
    p = M2.race_probs(booster.predict(X), meta["tau"])[0]
    if os.path.exists(os.path.join(DATA, "model3.json")):  # 2・3着はコースの形を学習した着順モデルで出す
        t3 = M3.Places().trifecta(feats, p)
    else:
        t3 = M2.harville(p, meta["gamma"])
    t2 = defaultdict(float)
    for k, v in t3.items():
        t2[k[:3]] += v
    odds3 = {k: float(v) for k, v in d["3連単オッズ"].items() if re.fullmatch(r"[\d.]+", v)}
    odds2 = {k: float(v) for k, v in d.get("2連単オッズ", {}).items() if re.fullmatch(r"[\d.]+", v)}
    names = {b["枠"]: b["選手"] for b in d["出走表"]["艇"]}
    top3 = defaultdict(float)
    for k, v in t3.items():
        for b in k.split("-"):
            top3[int(b)] += v
    out = [f"# {d['場']} {rno}R ボート基盤モデル v2（{cat}）", "",
           "| 枠 | 選手 | 1着確率 | 3着以内確率 | そのコースの過去1年1着率 | モーターの最近の平均着点 | 今節の平均着点 |",
           "|---|---|---|---|---|---|---|"]
    for i in range(6):
        f = feats[i + 1]
        fmt = lambda v, n=1: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"
        out.append(f"| {i + 1} | {names[i + 1]} | {100 * p[i]:.1f}% | {100 * top3[i + 1]:.1f}% | "
                   f"{100 * f['rc_win']:.0f}%（{f['rc_n']}走） | {fmt(f['motor_pts'])} | {fmt(f['form_pts'])} |")
    out += ["", "## 2連単（確率順）", "| 買い目 | 確率 | オッズ | 期待値 |", "|---|---|---|---|"]
    for k in sorted(t2, key=t2.get, reverse=True)[:8]:
        o = odds2.get(k)
        out.append(f"| {k} | {100 * t2[k]:.1f}% | {o or '-'} | {(t2[k] * o):.2f} |" if o else f"| {k} | {100 * t2[k]:.1f}% | - | - |")
    out += ["", "## 3連単（確率順）", "| 買い目 | 確率 | オッズ | 期待値 |", "|---|---|---|---|"]
    for k in sorted(t3, key=t3.get, reverse=True)[:10]:
        o = odds3.get(k)
        out.append(f"| {k} | {100 * t3[k]:.1f}% | {o or '-'} | {(t3[k] * o):.2f} |" if o else f"| {k} | {100 * t3[k]:.1f}% | - | - |")
    go, cands, hit, reasons = P.verdict(p, t3, t2, odds3, odds2, budget)
    out += ["", f"## 判定：{'✅ 買い' if go else '⛔ 見送り推奨'}（予算{budget}円）"]
    out += [f"- {r}" for r in reasons]
    if cands:
        out += ["", "| 券種 | 買い目 | 確率（モデルと市場の平均） | オッズ | 期待値 |", "|---|---|---|---|---|"]
        out += [f"| {c[0]} | {c[1]} | {100 * c[2]:.1f}% | {c[3]} | {c[4]:.2f} |" for c in cands]
        out.append(f"\n- 合計の的中確率：約{100 * hit:.0f}%（各100円）")
    out.append("\n※ 当日のこれより前のレース結果は、v2 の履歴（今節の調子など）にはまだ入っていない")
    if log:
        P.log_prediction(jcd, rno, hd, p, cands, go)
    return "\n".join(out), p, t3


if __name__ == "__main__":
    argv = sys.argv[1:]
    opts = {}
    for name in ("--budget", "--cache"):
        if name in argv:
            j = argv.index(name)
            opts[name] = argv[j + 1]
            del argv[j:j + 2]
    print(predict(int(argv[0]), int(argv[1]), argv[2], int(opts.get("--budget", 500)),
                  opts.get("--cache", DEFAULT_CACHE))[0])
