#!/usr/bin/env python3
"""2着・3着モデル（v2 の1着確率と組み合わせて3連単・2連単の確率を出す）

v2 は2・3着を「1着確率のγ乗」（Harville）で出していたため、
「1号艇が逃げたら2着は2コースが残りやすい」というコースの形が入らず、
1-5・1-6 を高く、1-2 を低く見積もっていた（knowledge.md 13-7）。

ここでは
- 2着モデル：1着の艇が決まったとき、残り5艇のどれが2着か
- 3着モデル：1・2着が決まったとき、残り4艇のどれが3着か
を、1着の艇・2着の艇のコースとの位置関係を特徴量に入れて直接学習する。
3連単の確率 = v2の1着確率 × 2着モデル × 3着モデル

期間の分け方は v2 と同じ（学習 2023-10〜2025-09 ／ 調整 2025-10〜2026-03 ／ テスト 2026-04〜09）

使い方: python3 model3.py <model2_data.pkl> [--save ../data/model3]
"""
import itertools
import json
import os
import pickle
import sys
from datetime import date

import lightgbm as lgb
import numpy as np

import model2 as M2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CTX2 = ["w_course", "rel_w", "w_nat", "w_ex", "nat_rank_rest"]
CTX3 = CTX2 + ["s_course", "rel_s", "s_nat", "between"]
FEAT2 = M2.FEATURES + CTX2
FEAT3 = M2.FEATURES + CTX3
CAT2 = M2.CATEGORICAL + ["w_course"]
CAT3 = CAT2 + ["s_course"]


def ctx_row(fs, j, w, s=None):
    """候補の艇 j の特徴量に、1着 w（と2着 s）との関係を足す"""
    f = fs[j]
    rest = [b for b in fs if b != w and b != s]
    nat_rest = sorted(rest, key=lambda b: -np.nan_to_num(fs[b]["nat"]))
    row = [f[k] for k in M2.FEATURES]
    row += [fs[w]["course"], f["course"] - fs[w]["course"], fs[w]["nat"], fs[w]["ex_diff"], nat_rest.index(j) + 1]
    if s is not None:
        lo, hi = sorted((fs[w]["course"], fs[s]["course"]))
        row += [fs[s]["course"], f["course"] - fs[s]["course"], fs[s]["nat"], 1 if lo < f["course"] < hi else 0]
    return row


def xy(data, stage):
    """学習用：実際の1着（と2着）を条件にした行"""
    X, y = [], []
    for r in data:
        o = r["order"]
        if len(o) < stage + 1:
            continue
        w, s = o[0], (o[1] if stage == 3 else None)
        for j in range(1, 7):
            if j == w or j == s:
                continue
            X.append(ctx_row(r["feats"], j, w, s))
            y.append(1 if j == o[stage - 1] else 0)
    return np.array(X, dtype=float), np.array(y)


def group_softmax(raw, n, tau):
    p = raw.reshape(-1, n)
    lo = np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(np.clip(1 - p, 1e-6, 1))
    s = tau * lo
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def train(Xtr, ytr, Xva, yva, feats, cats):
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1}
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, categorical_feature=cats)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    return lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


class Places:
    """v2 の1着確率 p と2着・3着モデルから、3連単の全120通りの確率を出す"""

    def __init__(self, path=os.path.join(DATA, "model3")):
        self.b2 = lgb.Booster(model_file=path + "_2nd.txt")
        self.b3 = lgb.Booster(model_file=path + "_3rd.txt")
        self.meta = json.load(open(path + ".json"))

    def trifecta(self, fs, p):
        X2 = [ctx_row(fs, j, w) for w in range(1, 7) for j in range(1, 7) if j != w]
        q2 = group_softmax(self.b2.predict(np.array(X2, dtype=float)), 5, self.meta["tau2"])
        keys3, X3 = [], []
        for w, s in itertools.permutations(range(1, 7), 2):
            keys3.append((w, s))
            X3 += [ctx_row(fs, j, w, s) for j in range(1, 7) if j not in (w, s)]
        q3 = group_softmax(self.b3.predict(np.array(X3, dtype=float)), 4, self.meta["tau3"])
        p2 = {}
        for i, w in enumerate(range(1, 7)):
            for jj, j in enumerate([b for b in range(1, 7) if b != w]):
                p2[(w, j)] = q2[i][jj]
        t3 = {}
        for i, (w, s) in enumerate(keys3):
            for kk, k in enumerate([b for b in range(1, 7) if b not in (w, s)]):
                t3[f"{w}-{s}-{k}"] = float(p[w - 1] * p2[(w, s)] * q3[i][kk])
        return t3


def main(argv):
    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            return getattr(M2, name) if name == "History" else super().find_class(module, name)

    data, _ = _U(open(argv[0], "rb")).load()
    tr = [r for r in data if r["date"] < date(2025, 10, 1)]
    va = [r for r in data if date(2025, 10, 1) <= r["date"] < date(2026, 4, 1)]
    te = [r for r in data if r["date"] >= date(2026, 4, 1)]
    out = {}
    models = {}
    for stage, n, feats, cats in ((2, 5, FEAT2, CAT2), (3, 4, FEAT3, CAT3)):
        Xtr, ytr = xy(tr, stage)
        Xva, yva = xy(va, stage)
        b = train(Xtr, ytr, Xva, yva, feats, cats)
        raw = b.predict(Xva, num_iteration=b.best_iteration)
        ll = lambda t: float(-np.mean(np.log(group_softmax(raw, n, t)[yva.reshape(-1, n) == 1])))
        tau = min([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], key=ll)
        out[f"tau{stage}"] = tau
        models[stage] = b
        print(f"{stage}着モデル：反復{b.best_iteration}、τ={tau}、調整期間の対数損失 {ll(tau):.4f}（でたらめなら {np.log(n):.4f}）")
    if "--save" in argv:
        path = argv[argv.index("--save") + 1]
        models[2].save_model(path + "_2nd.txt", num_iteration=models[2].best_iteration)
        models[3].save_model(path + "_3rd.txt", num_iteration=models[3].best_iteration)
        json.dump(out, open(path + ".json", "w"))
    return te


if __name__ == "__main__":
    main(sys.argv[1:])
