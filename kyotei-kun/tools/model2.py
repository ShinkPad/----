#!/usr/bin/env python3
"""ボート基盤モデル v2（LightGBM ＋ 過去成績から作る特徴量）

v1（model.py、条件付きロジット）からの改善点
- 学習データを3年分に拡大（2023-10〜2026-09）
- 「その時点より前の結果だけ」から作る履歴特徴量（未来の情報を使わない）
  - 選手：そのコースでの過去1年の1着率・3着内率、直近90日の平均着点、直近の平均ST
  - モーター：その場のそのモーター番号の直近120日の平均着点（今の機力）
  - 今節（直近6日・同じ場）の平均着点
- レースの中での相対値（全国勝率の順位、内側の艇の強さ、1号艇の強さ など）
- 場・コース・風向をカテゴリとして扱い、場ごとのクセを学習
- 非線形の決定木モデル（LightGBM）で、組み合わせの効果を自動で学習
- 2・3着は「1着確率のγ乗」で補正したHarville（γは検証期間で決める）

期間の分け方：学習 2023-10〜2025-09 ／ 調整 2025-10〜2026-03 ／ 最終テスト 2026-04〜2026-09

使い方: python3 model2.py <キャッシュ> [--save ../data/model2.txt]
"""
import itertools
import json
import os
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np

import backtest as B
import stats as S

POINTS = {"01": 10, "02": 8, "03": 6, "04": 4, "05": 2, "06": 1}
CAT_ORDER = ["予選", "一般戦", "特選・選抜", "その他（場独自の企画名）", "準優勝戦", "ドリーム戦", "優勝戦"]
WINDS = ["無風", "北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", "南", "南南西", "南西", "西南西", "西",
         "西北西", "北西", "北北西"]
FEATURES = [
    "venue", "course", "waku", "cat", "wind_dir", "wind", "wave",
    "nat", "nat2", "loc", "loc_none", "cls", "age", "weight", "mot2", "boat2",
    "ex_diff", "ex_rank", "nat_rank", "nat_gap_top", "nat_inner", "ex_inner", "nat_b1", "ex_b1",
    "rc_n", "rc_win", "rc_top3", "rec_pts", "rec_n", "st_avg", "st_n",
    "motor_pts", "motor_n", "form_pts", "form_n",
]
CATEGORICAL = ["venue", "course", "waku", "cat", "wind_dir"]


class History:
    """日付順に結果を流し込み、ある時点より前の成績だけを返す"""

    def __init__(self):
        self.rc = defaultdict(deque)      # (登番, コース) -> (日付, 1着, 3着内)
        self.rec = defaultdict(deque)     # 登番 -> (日付, 着点)
        self.st = defaultdict(deque)      # 登番 -> (日付, ST)
        self.motor = defaultdict(deque)   # (場, モーター) -> (日付, 着点)
        self.form = defaultdict(deque)    # (場, 登番) -> (日付, 着点)

    @staticmethod
    def _prune(q, d, days):
        while q and (d - q[0][0]).days > days:
            q.popleft()

    def feats(self, d, toban, course, venue, motor):
        q = self.rc[(toban, course)]
        self._prune(q, d, 365)
        n = len(q)
        win = sum(x[1] for x in q)
        top3 = sum(x[2] for x in q)
        base_win = [0.55, 0.14, 0.13, 0.10, 0.06, 0.02][course - 1]
        base_top3 = [0.81, 0.55, 0.53, 0.50, 0.38, 0.22][course - 1]
        r = self.rec[toban]
        self._prune(r, d, 90)
        s = self.st[toban]
        self._prune(s, d, 90)
        m = self.motor[(venue, motor)]
        self._prune(m, d, 120)
        f = self.form[(venue, toban)]
        self._prune(f, d, 6)
        return {
            "rc_n": n, "rc_win": (win + 5 * base_win) / (n + 5), "rc_top3": (top3 + 5 * base_top3) / (n + 5),
            "rec_pts": (sum(x[1] for x in r) / len(r)) if r else np.nan, "rec_n": len(r),
            "st_avg": (sum(x[1] for x in s) / len(s)) if s else np.nan, "st_n": len(s),
            "motor_pts": (sum(x[1] for x in m) / len(m)) if m else np.nan, "motor_n": len(m),
            "form_pts": (sum(x[1] for x in f) / len(f)) if f else np.nan, "form_n": len(f),
        }

    def update(self, d, venue, rows):
        for x in rows:
            if not x.get("登番") or not x.get("進入"):
                continue
            pos = x["着"]
            pts = POINTS.get(pos, 0)
            self.rc[(x["登番"], x["進入"])].append((d, 1 if pos == "01" else 0, 1 if pos in ("01", "02", "03") else 0))
            self.rec[x["登番"]].append((d, pts))
            self.form[(venue, x["登番"])].append((d, pts))
            if x.get("モーター"):
                self.motor[(venue, x["モーター"])].append((d, pts))
            st = x.get("ST") or ""
            try:
                v = float(st)
                if 0 <= v < 1:
                    self.st[x["登番"]].append((d, v))
            except ValueError:
                pass


def race_rows(d, r, bo, hist):
    """1レース分の6艇の特徴量（結果は使わない）"""
    rows = {x["艇"]: x for x in r["艇"]}
    ex_t = {b: rows[b]["展示"] for b in rows if rows[b]["展示"]}
    if len(ex_t) < 6 or len(bo) < 6 or len(rows) < 6:
        return None
    course = {b: rows[b]["進入"] or b for b in rows}
    mean_ex = sum(ex_t.values()) / 6
    ex_rank = B.ranks({b: -ex_t[b] for b in rows}, True)
    nat = {b: bo[b]["全国勝率"] for b in rows}
    nat_rank = B.ranks(nat, True)
    by_course = {course[b]: b for b in rows}
    cat = S.race_category(r["種別"])
    wd = r["風向"] if r["風向"] in WINDS else "無風"
    out = {}
    for b in range(1, 7):
        x = bo[b]
        c = course[b]
        inner = by_course.get(c - 1)
        f = {
            "venue": r["場"], "course": c, "waku": b, "cat": CAT_ORDER.index(cat) if cat in CAT_ORDER else 3,
            "wind_dir": WINDS.index(wd), "wind": r["風速"], "wave": r["波高"],
            "nat": nat[b], "nat2": x["全国2率"], "loc": x["当地勝率"] if x["当地勝率"] > 0 else np.nan,
            "loc_none": 1 if x["当地勝率"] == 0 else 0, "cls": {"A1": 0, "A2": 1, "B1": 2, "B2": 3}[x["級別"]],
            "age": x["年齢"], "weight": x["体重"], "mot2": x["モーター2率"], "boat2": x["ボート2率"] or np.nan,
            "ex_diff": mean_ex - ex_t[b], "ex_rank": ex_rank[b], "nat_rank": nat_rank[b],
            "nat_gap_top": nat[b] - max(v for k, v in nat.items() if k != b),
            "nat_inner": nat[inner] if inner else np.nan, "ex_inner": (mean_ex - ex_t[inner]) if inner else np.nan,
            "nat_b1": nat[by_course.get(1, 1)], "ex_b1": mean_ex - ex_t[by_course.get(1, 1)],
        }
        f.update(hist.feats(d, x["登番"], c, r["場"], rows[b].get("モーター")))
        out[b] = f
    return out


def build(start, end, cache, hist=None):
    """start〜end の各レースの特徴量を作る。hist を渡すと、その続きから履歴を積み上げる（最後の hist も返す）"""
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    hist = hist or History()
    data = []
    with ThreadPoolExecutor(8) as ex:
        for d, (races, prog) in zip(days, ex.map(lambda x: B.load(x, cache), days)):
            for r in sorted(races, key=lambda r: (r["場"], r["R"])):
                bo = prog.get((r["場"], r["R"]), {}).get("艇", {})
                if len(r["艇"]) == 6 and r["3連単配当"] and any(x["着"] == "01" for x in r["艇"]):
                    fs = race_rows(d, r, bo, hist)
                    if fs:
                        order = [x["艇"] for x in sorted(r["艇"], key=lambda x: x["着"]) if x["着"] in POINTS]
                        data.append({"date": d, "feats": fs, "order": order, "t3": r["3連単"], "p3": r["3連単配当"],
                                     "t2": r.get("2連単"), "p2": r.get("2連単配当"), "win": r.get("単勝"),
                                     "pw": r.get("単勝配当")})
                hist.update(d, r["場"], r["艇"])
    build.hist = hist
    return data


def to_xy(data):
    X = np.array([[r["feats"][b][k] for k in FEATURES] for r in data for b in range(1, 7)], dtype=float)
    y = np.array([1 if r["order"][0] == b else 0 for r in data for b in range(1, 7)])
    return X, y


def race_probs(raw, tau=1.0):
    p = raw.reshape(-1, 6)
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1))
    s = tau * lo
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def logloss(P, data):
    y = np.array([r["order"][0] - 1 for r in data])
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def harville(p, gamma=1.0):
    q = p ** gamma
    q = q / q.sum()
    res = {}
    for a, b, c in itertools.permutations(range(6), 3):
        res[f"{a + 1}-{b + 1}-{c + 1}"] = p[a] * q[b] / (1 - q[a]) * q[c] / (1 - q[a] - q[b])
    return res


def exacta_ll(P, data, gamma):
    ll = []
    for p, r in zip(P, data):
        if len(r["order"]) < 3:
            continue
        t = harville(p, gamma)
        ll.append(-np.log(t[f"{r['order'][0]}-{r['order'][1]}-{r['order'][2]}"] + 1e-12))
    return float(np.mean(ll))


def evaluate(P, data, gamma, label):
    hit = defaultdict(lambda: [0, 0, 0])
    for p, r in zip(P, data):
        t = harville(p, gamma)
        top = sorted(t, key=t.get, reverse=True)
        ex2 = defaultdict(float)
        for k, v in t.items():
            ex2[k[:3]] += v
        top2 = sorted(ex2, key=ex2.get, reverse=True)
        best = int(np.argmax(p)) + 1
        for name, picks, key, pay in [
            ("単勝 1位", {best}, r["win"], r["pw"] or 0),
            ("2連単 上位1点", set(top2[:1]), r["t2"], r["p2"] or 0),
            ("2連単 上位2点", set(top2[:2]), r["t2"], r["p2"] or 0),
            ("3連単 上位4点", set(top[:4]), r["t3"], r["p3"]),
            ("3連単 上位6点", set(top[:6]), r["t3"], r["p3"]),
            ("3連単 上位10点", set(top[:10]), r["t3"], r["p3"]),
        ]:
            a = hit[name]
            a[0] += 1 if key in picks else 0
            a[1] += 100 * len(picks)
            a[2] += pay if key in picks else 0
    n = len(data)
    print(f"\n### {label}（{n}レース）")
    print("| 買い方 | 的中率 | 回収率 |")
    print("|---|---|---|")
    for k, (h, inv, pay) in hit.items():
        print(f"| {k} | {100 * h / n:.1f}% | {100 * pay / inv:.1f}% |")


def main(argv):
    save = None
    if "--save" in argv:
        i = argv.index("--save")
        save = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    cache = argv[0]
    import pickle
    pk = os.path.join(cache, "model2_data.pkl")
    if os.path.exists(pk):
        data, hist = pickle.load(open(pk, "rb"))
    else:
        data = build(date(2023, 10, 1), date(2026, 9, 30), cache)
        hist = build.hist
        pickle.dump((data, hist), open(pk, "wb"))
    tr = [r for r in data if r["date"] < date(2025, 10, 1)]
    va = [r for r in data if date(2025, 10, 1) <= r["date"] < date(2026, 4, 1)]
    te = [r for r in data if r["date"] >= date(2026, 4, 1)]
    print(f"学習 {len(tr)} ／ 調整 {len(va)} ／ テスト {len(te)} レース")
    Xtr, ytr = to_xy(tr)
    Xva, yva = to_xy(va)
    Xte, yte = to_xy(te)
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
    dtr = lgb.Dataset(Xtr, ytr, feature_name=FEATURES, categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    booster = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"木の数 {booster.best_iteration}")
    Pva_raw = booster.predict(Xva, num_iteration=booster.best_iteration)
    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    tau = min(taus, key=lambda t: logloss(race_probs(Pva_raw, t), va))
    Pva = race_probs(Pva_raw, tau)
    gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    gamma = min(gammas, key=lambda g: exacta_ll(Pva[:4000], va[:4000], g))
    print(f"τ={tau}, γ={gamma}")
    Pte = race_probs(booster.predict(Xte, num_iteration=booster.best_iteration), tau)
    yt = np.array([r["order"][0] - 1 for r in te])
    base = np.tile([0.558, 0.131, 0.128, 0.102, 0.061, 0.019], (len(te), 1))
    print(f"\n## テスト期間の精度")
    print(f"- 基準（コースの平均）logloss {logloss(base, te):.4f}")
    print(f"- v2 logloss {logloss(Pte, te):.4f}、1着的中率 {100 * np.mean(Pte.argmax(1) == yt):.1f}%")
    print(f"- 3連単の正解の平均確率（対数）：v2 {exacta_ll(Pte, te, gamma):.3f}")
    wk = [i for i, r in enumerate(te) if all(r["feats"][b]["course"] == b for b in range(1, 7))]
    print(f"- 枠なりのレースだけ（v1と同じ条件）：logloss {logloss(Pte[wk], [te[i] for i in wk]):.4f}、"
          f"1着的中率 {100 * np.mean(Pte[wk].argmax(1) == yt[wk]):.1f}%（v1は 1.1448 / 58.9%）")
    imp = sorted(zip(FEATURES, booster.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(v for _, v in imp)
    print("\n## 効いている特徴量（寄与の割合）")
    for k, v in imp:
        print(f"- {k}: {100 * v / tot:.1f}%")
    evaluate(Pte, te, gamma, "テスト期間で買った場合（実際の払戻）")
    if save:
        booster.save_model(save, num_iteration=booster.best_iteration)
        with open(save + ".json", "w") as f:
            json.dump({"tau": tau, "gamma": gamma, "features": FEATURES, "history_until": "2026-09-30"}, f)
        import gzip
        pickle.dump(hist, gzip.open(os.path.join(os.path.dirname(save), "model2_hist_base.pkl.gz"), "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
