#!/usr/bin/env python3
"""選手の走り方のタイプ分け（教師なし学習）

1. 選手ごとに、そのレースより前の1年間の「走り方のクセ」を数える（未来の結果は使わない）
   - 逃げ率（1コースで逃げて勝った割合）
   - まくり率（3〜6コースから、まくり・まくり差しで勝った割合）
   - 差し率（2〜6コースから、差しで勝った割合）
   - 外コースの3着内率（4〜6コース）
   - STの平均とばらつき
   - 前付けの傾向（枠番 − 進入コース の平均。プラスなら内に入りたがる）
2. 学習期間の最後の時点のクセを標準化して k-means でタイプに分ける（正解の着順は使わない）
3. 予想するときは、その時点のクセから一番近いタイプを割り当てる

使い方: python3 style.py <キャッシュ>   … 学習期間（〜2025-09）でタイプを作り、data/style.json に保存
"""
import json
import os
import sys
from collections import defaultdict, deque
from datetime import date, timedelta

import numpy as np

import backtest as B

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FEATS = ["nige", "makuri", "sashi", "out_top3", "st_mean", "st_sd", "maezuke", "mk_share", "ms_share", "sa_share"]
# タイプ分けには実力に左右されにくい指標だけを使う（実力は v2 が勝率などで既に見ている）
DIMS = ["mk_share", "ms_share", "sa_share", "st_mean", "st_sd", "maezuke"]
# 数が少ない選手を平均に寄せるための事前値（全体の平均くらい）
PRIOR = {"nige": (0.55, 10), "makuri": (0.06, 20), "sashi": (0.05, 20), "out_top3": (0.33, 20),
         "st_mean": (0.16, 10), "st_sd": (0.05, 10), "maezuke": (0.0, 20),
         "mk_share": (0.30, 8), "ms_share": (0.30, 8), "sa_share": (0.30, 8)}
N = 14


class Style:
    """日付順に結果を流し込み、ある時点より前の1年間のクセを返す（足し引きで高速に更新）"""

    def __init__(self, days=365):
        self.days = days
        self.q = defaultdict(deque)       # 登番 -> (日付, 値のタプル)
        self.s = defaultdict(lambda: np.zeros(N))

    def _prune(self, toban, d):
        q = self.q[toban]
        while q and (d - q[0][0]).days > self.days:
            self.s[toban] -= q.popleft()[1]

    def profile(self, d, toban):
        self._prune(toban, d)
        s = self.s[toban]
        n1, nige, n36, mak, n26, sas, n46, t46, nst, st, st2, ow, ms, mk = s
        def sm(k, num, den):
            m, w = PRIOR[k]
            return (num + m * w) / (den + w)
        st_mean = sm("st_mean", st, nst)
        var = (st2 / nst - (st / nst) ** 2) if nst >= 5 else PRIOR["st_sd"][0] ** 2
        return {"nige": sm("nige", nige, n1), "makuri": sm("makuri", mak, n36), "sashi": sm("sashi", sas, n26),
                "out_top3": sm("out_top3", t46, n46), "st_mean": st_mean,
                "st_sd": sm("st_sd", np.sqrt(max(var, 0)) * nst, nst), "maezuke": self.mz(toban),
                # 2〜6コースから勝ったときの勝ち方の内訳（まくり／まくり差し／差し）
                "mk_share": sm("mk_share", mk, ow), "ms_share": sm("ms_share", ms, ow), "sa_share": sm("sa_share", sas, ow),
                "n": int(len(self.q[toban]))}

    def mz(self, toban):
        q = self.q[toban]
        if not q:
            return 0.0
        m, w = PRIOR["maezuke"]
        return (sum(x[2] for x in q) + m * w) / (len(q) + w)

    def update(self, d, kimarite, rows):
        for x in rows:
            tb, c, waku, pos = x.get("登番"), x.get("進入"), x.get("艇"), x.get("着")
            if not tb or not c:
                continue
            win = pos == "01"
            v = np.zeros(N)
            v[0] = c == 1
            v[1] = c == 1 and win and kimarite == "逃げ"
            v[2] = c >= 3
            v[3] = c >= 3 and win and kimarite in ("まくり", "まくり差し")
            v[4] = c >= 2
            v[5] = c >= 2 and win and kimarite == "差し"
            v[6] = c >= 4
            v[7] = c >= 4 and pos in ("01", "02", "03")
            v[11] = c >= 2 and win
            v[12] = c >= 2 and win and kimarite == "まくり差し"
            v[13] = c >= 2 and win and kimarite == "まくり"
            try:
                st = float(x.get("ST") or "")
                if 0 <= st < 1:
                    v[8], v[9], v[10] = 1, st, st * st
            except ValueError:
                pass
            self.q[tb].append((d, v, waku - c))
            self.s[tb] += v


class Types:
    """k-means で作ったタイプ（中心）を使って、クセからタイプを割り当てる"""

    def __init__(self, path=os.path.join(DATA, "style.json")):
        m = json.load(open(path))
        self.mu, self.sd = np.array(m["mu"]), np.array(m["sd"])
        self.centers = np.array(m["centers"])
        self.names = m["names"]

    def assign(self, prof):
        z = (np.array([prof[k] for k in DIMS]) - self.mu) / self.sd
        return int(np.argmin(((self.centers - z) ** 2).sum(1)))


def kmeans(Z, k, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), 1)]
    for _ in range(1, k):  # k-means++ の初期化
        d = ((Z[:, None, :] - C[None]) ** 2).sum(2).min(1)
        C = np.vstack([C, Z[rng.choice(len(Z), 1, p=d / d.sum())]])
    for _ in range(iters):
        lab = ((Z[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        C2 = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C, C2):
            break
        C = C2
    inertia = ((Z - C[lab]) ** 2).sum()
    return C, lab, inertia


def silhouette(Z, lab, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    Zs, ls = Z[idx], lab[idx]
    D = np.sqrt(((Zs[:, None] - Zs[None]) ** 2).sum(2))
    s = []
    for i in range(len(Zs)):
        same = D[i, ls == ls[i]]
        a = same.sum() / max(len(same) - 1, 1)
        b = min(D[i, ls == j].mean() for j in set(ls) if j != ls[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def name_type(c):
    """タイプの中心（標準化した値）から、分かりやすい名前をつける"""
    d = dict(zip(DIMS, c))
    tags = []
    if d["mk_share"] > 0.6:
        tags.append("まくり型")
    if d["ms_share"] > 0.6:
        tags.append("まくり差し型")
    if d["sa_share"] > 0.6:
        tags.append("差し型")
    if d["st_mean"] < -0.7:
        tags.append("スタート速い")
    elif d["st_mean"] > 0.7:
        tags.append("スタート遅め")
    if d["st_sd"] > 0.7:
        tags.append("ST不安定")
    if d["maezuke"] > 0.7:
        tags.append("前付け")
    return "・".join(tags) or "平均型"


def main(argv):
    cache = argv[0]
    st = Style()
    d, end = date(2024, 10, 1), date(2025, 9, 30)
    for day in [date(2023, 10, 1) + timedelta(n) for n in range((d - date(2023, 10, 1)).days)]:
        for r in B.load(day, cache)[0]:
            st.update(day, r["決まり手"], r["艇"])
    while d <= end:
        for r in B.load(d, cache)[0]:
            st.update(d, r["決まり手"], r["艇"])
        d += timedelta(1)
    profs = [(tb, st.profile(end, tb)) for tb in list(st.q)]
    profs = [(tb, p) for tb, p in profs if p["n"] >= 50]
    X = np.array([[p[k] for k in DIMS] for _, p in profs])
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    print(f"選手 {len(Z)} 人（学習期間の最後の1年に50走以上）")
    best = None
    for k in range(4, 11):
        C, lab, inertia = min((kmeans(Z, k, s) for s in range(5)), key=lambda x: x[2])
        sil = silhouette(Z, lab)
        print(f"- タイプ数 {k}：まとまりの良さ {sil:.3f}")
        if best is None or sil > best[0]:
            best = (sil, k, C, lab)
    sil, k, C, lab = best
    names = [name_type(c) for c in C]
    print(f"\nタイプ数 {k} を採用")
    for j in range(k):
        raw = C[j] * sd + mu
        print(f"- タイプ{j}（{names[j]}、{(lab == j).sum()}人）：" +
              "、".join(f"{a} {b:.3f}" for a, b in zip(DIMS, raw)))
    json.dump({"dims": DIMS, "mu": mu.tolist(), "sd": sd.tolist(), "centers": C.tolist(), "names": names, "k": k},
              open(os.path.join(DATA, "style.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
