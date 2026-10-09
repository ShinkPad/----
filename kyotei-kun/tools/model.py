#!/usr/bin/env python3
"""ボート基盤モデル（条件付きロジット＝6艇の中で誰が1着かを確率で出す）

- 学習：前半期間、検証：後半期間（たまたまの当てはまりを避ける）
- 2着・3着は「残った艇の中で同じ強さの比率」で順に計算（Harville）
- 特徴量のグループを1つずつ外して（アブレーション）、どの判断要素が効いているかを測る
- 実際の払戻で、モデルの上位の買い目の回収率も測る

使い方:
  python3 model.py <開始YYYYMMDD> <終了YYYYMMDD> <分割YYYYMMDD> <キャッシュ> [--save 重みJSON]
"""
import itertools
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np

import backtest as B
import stats as S

POINTS = {"01": 10, "02": 8, "03": 6, "04": 4, "05": 2, "06": 1}


def build(start, end, cache):
    """レースごとに (日付, 6艇の特徴量dict, 着順の枠リスト, 払戻) を作る"""
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    form = defaultdict(list)  # (場, 登番) -> [(日付, 着点)] 今節の調子
    out = []
    with ThreadPoolExecutor(8) as ex:
        for d, (races, prog) in zip(days, ex.map(lambda x: B.load(x, cache), days)):
            races = sorted(races, key=lambda r: (r["場"], r["R"]))
            daytrend = defaultdict(lambda: [0] * 7)  # 場 -> コース別1着数（その日のそれまで）
            daycount = defaultdict(int)
            for r in races:
                bo = prog.get((r["場"], r["R"]), {}).get("艇", {})
                rows = {x["艇"]: x for x in r["艇"]}
                ok = (len(rows) == 6 and len(bo) == 6 and r["3連単配当"] and r.get("2連単配当")
                      and all(rows[b]["進入"] == b for b in rows) and all(rows[b]["展示"] for b in rows))
                if ok:
                    ex_t = {b: rows[b]["展示"] for b in rows}
                    mean_ex = sum(ex_t.values()) / 6
                    ex_rank = B.ranks({b: -ex_t[b] for b in rows}, True)
                    nat = {b: bo[b]["全国勝率"] for b in rows}
                    nat_mean = sum(nat.values()) / 6
                    mot = {b: bo[b]["モーター2率"] for b in rows}
                    mot_rank = B.ranks(mot, True)
                    cat = S.race_category(r["種別"])
                    feats = {}
                    for b in range(1, 7):
                        x = bo[b]
                        hist = [p for (dd, p) in form[(r["場"], x["登番"])] if (d - dd).days <= 6]
                        tr = daytrend[r["場"]]
                        n_before = daycount[r["場"]]
                        f = {
                            "course": b,
                            "nat": nat[b] - nat_mean,
                            "loc": (x["当地勝率"] - nat[b]) if x["当地勝率"] > 0 else 0.0,
                            "loc_none": 1.0 if x["当地勝率"] == 0 else 0.0,
                            "nat2": (x["全国2率"] - 35) / 10,
                            "cls": {"A1": 0, "A2": 1, "B1": 2, "B2": 3}[x["級別"]],
                            "mot": (mot[b] - 35) / 10,
                            "mot_top": 1.0 if mot_rank[b] == 1 else 0.0,
                            "boat": ((x["ボート2率"] or 35) - 35) / 10,
                            "ex": (mean_ex - ex_t[b]) * 10,
                            "ex_top": 1.0 if ex_rank[b] == 1 else 0.0,
                            "weight": (x["体重"] - 52) / 3,
                            "age": (x["年齢"] - 40) / 10,
                            "form": (sum(hist) / len(hist) - 5.5) / 3 if hist else 0.0,
                            "form_n": min(len(hist), 8) / 8,
                            "wind": r["風速"], "wave": r["波高"],
                            "trend": (tr[b] / n_before) if n_before else 0.0,
                            "final": 1.0 if cat in ("優勝戦",) else 0.0,
                            "seed": 1.0 if cat in ("優勝戦", "ドリーム戦", "特選・選抜", "その他（場独自の企画名）") else 0.0,
                            "yosen": 1.0 if cat == "予選" else 0.0,
                        }
                        feats[b] = f
                    order = sorted(rows, key=lambda b: rows[b]["着"])
                    order = [b for b in order if rows[b]["着"] in POINTS]
                    out.append({"date": d, "place": r["場"], "R": r["R"], "feats": feats, "order": order,
                                "t3": r["3連単"], "p3": r["3連単配当"], "t2": r["2連単"], "p2": r["2連単配当"],
                                "win": r.get("単勝"), "pw": r.get("単勝配当")})
                # 今節の調子・当日の流れを更新（結果を見た後）
                for x in r["艇"]:
                    if x.get("登番") and x["着"] in POINTS:
                        form[(r["場"], x["登番"])].append((d, POINTS[x["着"]]))
                win = next((x for x in r["艇"] if x["着"] == "01"), None)
                if win and win["進入"]:
                    daytrend[r["場"]][win["進入"]] += 1
                daycount[r["場"]] += 1
    return out


GROUPS = {
    "コース": lambda f: [1.0 if f["course"] == c else 0.0 for c in range(2, 7)],
    "全国勝率": lambda f: [f["nat"], f["nat"] * (f["course"] == 1), f["nat2"]],
    "級別": lambda f: [1.0 if f["cls"] == k else 0.0 for k in (1, 2, 3)],
    "当地勝率": lambda f: [f["loc"], f["loc_none"]],
    "モーター": lambda f: [f["mot"], f["mot_top"]],
    "ボート": lambda f: [f["boat"]],
    "展示タイム": lambda f: [f["ex"], f["ex_top"], f["ex"] * (f["course"] == 1)],
    "体重・年齢": lambda f: [f["weight"], f["age"]],
    "今節の調子": lambda f: [f["form"] * f["form_n"]],
    "風・波×コース": lambda f: [f["wind"] * (f["course"] == 1), f["wind"] * (f["course"] in (3, 4)),
                                f["wave"] * (f["course"] == 1)],
    "レース種別×コース": lambda f: [f["seed"] * (f["course"] == 1), f["yosen"] * (f["course"] == 1),
                                  f["final"] * (f["course"] == 1)],
    "当日の流れ": lambda f: [f["trend"]],
}


def matrix(data, groups):
    X = np.array([[sum((GROUPS[g](r["feats"][b]) for g in groups), []) for b in range(1, 7)] for r in data])
    return X


def fit(X, y, l2=1e-4, iters=600, lr=0.05):
    """特徴量を標準化して Adam で学習。戻り値は (重み, 平均, 標準偏差)"""
    n, k, d = X.shape
    mu = X.reshape(-1, d).mean(0)
    sd = X.reshape(-1, d).std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    Y = np.zeros((n, k))
    Y[np.arange(n), y] = 1
    for t in range(1, iters + 1):
        s = Z @ w
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        g = ((p - Y)[:, :, None] * Z).sum((0, 1)) / n + l2 * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return (w, mu, sd)


def probs(X, model):
    w, mu, sd = model
    s = ((X - mu) / sd) @ w
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(1, keepdims=True)


def logloss(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def harville(p):
    """1着確率 → 3連単120通りの確率"""
    res = {}
    for a, b, c in itertools.permutations(range(6), 3):
        pa = p[a]
        pb = p[b] / (1 - p[a])
        pc = p[c] / (1 - p[a] - p[b])
        res[f"{a + 1}-{b + 1}-{c + 1}"] = pa * pb * pc
    return res


def betting(data, P):
    """モデルの確率上位の買い目を買った場合の成績"""
    rules = defaultdict(lambda: [0, 0, 0, 0])
    for r, p in zip(data, P):
        t = harville(p)
        top = sorted(t, key=t.get, reverse=True)
        ex2 = defaultdict(float)
        for k, v in t.items():
            ex2[k[:3]] += v
        top2 = sorted(ex2, key=ex2.get, reverse=True)
        best = int(np.argmax(p)) + 1

        def add(name, picks, hit_key, pay):
            a = rules[name]
            a[0] += 1
            a[2] += 100 * len(picks)
            if hit_key in picks:
                a[1] += 1
                a[3] += pay

        add("単勝 モデル1位", {best}, r["win"], r["pw"] or 0)
        add("2連単 モデル上位1点", set(top2[:1]), r["t2"], r["p2"])
        add("2連単 モデル上位2点", set(top2[:2]), r["t2"], r["p2"])
        add("2連単 モデル上位3点", set(top2[:3]), r["t2"], r["p2"])
        add("3連単 モデル上位4点", set(top[:4]), r["t3"], r["p3"])
        add("3連単 モデル上位6点", set(top[:6]), r["t3"], r["p3"])
        add("3連単 モデル上位10点", set(top[:10]), r["t3"], r["p3"])
        # 1着は固定、2・3着の相手を広げる（今日の負け方の対策）
        if best:
            others = [i + 1 for i in np.argsort(-p) if i + 1 != best]
            add("3連単 モデル1位-上位4艇-上位4艇（12点）", {f"{best}-{x}-{z}" for x, z in itertools.permutations(others[:4], 2)},
                r["t3"], r["p3"])
            add("2連単 モデル1位-全（5点）", {f"{best}-{x}" for x in others}, r["t2"], r["p2"])
    return rules


def main(argv):
    save = None
    if "--save" in argv:
        i = argv.index("--save")
        save = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    start, end, split = (datetime.strptime(x, "%Y%m%d").date() for x in argv[:3])
    cache = argv[3]
    data = build(start, end, cache)
    tr = [r for r in data if r["date"] < split and r["order"]]
    te = [r for r in data if r["date"] >= split and r["order"]]
    ytr = np.array([r["order"][0] - 1 for r in tr])
    yte = np.array([r["order"][0] - 1 for r in te])
    print(f"学習 {len(tr)} レース / 検証 {len(te)} レース")

    allg = list(GROUPS)
    base_p = np.array([[0.558, 0.131, 0.128, 0.102, 0.061, 0.019]] * len(te))
    print(f"\n基準（コースの平均1着率だけ）: 検証logloss {logloss(base_p, yte):.4f}")

    Xtr, Xte = matrix(tr, allg), matrix(te, allg)
    w = fit(Xtr, ytr)
    P = probs(Xte, w)
    full = logloss(P, yte)
    acc = float(np.mean(P.argmax(1) == yte))
    print(f"全部入りモデル: 検証logloss {full:.4f}、1着的中率 {100 * acc:.1f}%")

    print("\n## 判断要素を1つ外したときの悪化（大きいほど大事）")
    print("| 外した要素 | 検証logloss | 悪化 |")
    print("|---|---|---|")
    rows = []
    for g in allg:
        if g == "コース":
            continue
        gs = [x for x in allg if x != g]
        wg = fit(matrix(tr, gs), ytr)
        lg = logloss(probs(matrix(te, gs), wg), yte)
        rows.append((lg - full, g, lg))
    for dlt, g, lg in sorted(rows, reverse=True):
        print(f"| {g} | {lg:.4f} | {dlt * 1000:+.2f} |")

    # 重み
    names = []
    for g in allg:
        k = len(GROUPS[g](tr[0]["feats"][1]))
        names += [f"{g}[{i}]" for i in range(k)]
    print("\n## 重み")
    for n_, v in zip(names, w[0]):
        print(f"- {n_}: {v:+.3f}（標準化後）")

    print("\n## モデルで買った場合（検証期間・実際の払戻）")
    print("| 買い方 | レース数 | 的中率 | 回収率 |")
    print("|---|---|---|---|")
    for name, (n, h, inv, pay) in sorted(betting(te, P).items(), key=lambda x: -x[1][3] / x[1][2]):
        print(f"| {name} | {n} | {100 * h / n:.1f}% | {100 * pay / inv:.1f}% |")

    if save:
        with open(save, "w", encoding="utf-8") as f:
            json.dump({"groups": allg, "weights": list(map(float, w[0])), "mean": list(map(float, w[1])),
                       "std": list(map(float, w[2])), "names": names}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
