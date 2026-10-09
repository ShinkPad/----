#!/usr/bin/env python3
"""穴・イレギュラーの狙い目を探す。

各レース・各艇について「条件」を作り、その艇の単勝、その艇を頭にした2連単流し（5点）、
上位3艇のボックス・3連複などの回収率を、前半（学習）と後半（検証）に分けて集計する。
前半だけ良くて後半で崩れる条件は「たまたま」として扱う。

使い方: python3 patterns.py <開始YYYYMMDD> <終了YYYYMMDD> <分割YYYYMMDD> <キャッシュ>
"""
import itertools
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import backtest as B
import stats as S

CLS_ORDER = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}


def wind(v):
    return "風0-2m" if v <= 2 else "風3-4m" if v <= 4 else "風5m以上"


def main(argv):
    start, end, split = (datetime.strptime(x, "%Y%m%d").date() for x in argv[:3])
    cache = argv[3]
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    # key -> period -> [n, hit, invest, payout]
    win = defaultdict(lambda: {"学習": [0, 0, 0, 0], "検証": [0, 0, 0, 0]})
    ex2 = defaultdict(lambda: {"学習": [0, 0, 0, 0], "検証": [0, 0, 0, 0]})
    box = defaultdict(lambda: {"学習": [0, 0, 0, 0], "検証": [0, 0, 0, 0]})

    def add(table, key, period, hit, pay, pts):
        a = table[key][period]
        a[0] += 1
        a[2] += 100 * pts
        if hit:
            a[1] += 1
            a[3] += pay

    def load(d):
        return d, B.load(d, cache)

    with ThreadPoolExecutor(8) as ex:
        for d, (races, prog) in ex.map(load, days):
            period = "学習" if d < split else "検証"
            for r in races:
                if len(r["艇"]) < 6 or not r.get("単勝配当") or not r.get("2連単配当") or not r.get("3連複配当"):
                    continue
                if any(x["進入"] != x["艇"] for x in r["艇"]):
                    continue
                bo = prog.get((r["場"], r["R"]), {}).get("艇", {})
                ext = {x["艇"]: x["展示"] for x in r["艇"] if x["展示"]}
                if len(bo) < 6 or len(ext) < 6:
                    continue
                allb = range(1, 7)
                nat_rank = B.ranks({b: bo[b]["全国勝率"] for b in allb}, True)
                ex_rank = B.ranks({b: -ext[b] for b in allb}, True)
                mot_rank = B.ranks({b: bo[b]["モーター2率"] for b in allb}, True)
                cls1 = bo[1]["級別"]
                place = S.PLACES[r["場"] - 1]
                w = wind(r["風速"])
                first = int(r["2連単"][0])
                # ---- 各艇の条件 ----
                for b in range(2, 7):
                    c = bo[b]["級別"]
                    conds = [
                        f"{b}号艇・1号艇{cls1}・自分{c}",
                        f"{b}号艇・勝率{nat_rank[b]}位・展示{ex_rank[b]}位",
                        f"{b}号艇・1号艇{cls1}・自分の勝率{nat_rank[b]}位",
                        f"{b}号艇・展示1位・1号艇展示{ex_rank[1]}位" if ex_rank[b] == 1 else None,
                        f"{b}号艇・{w}・勝率{nat_rank[b]}位",
                        f"{b}号艇・{place}・勝率{nat_rank[b]}位",
                        f"{b}号艇・1号艇より格上（{cls1}→{c}）" if CLS_ORDER[c] < CLS_ORDER[cls1] else None,
                        f"{b}号艇・モーター1位・勝率{nat_rank[b]}位" if mot_rank[b] == 1 else None,
                    ]
                    for k in filter(None, conds):
                        add(win, k, period, r["単勝"] == b, r["単勝配当"], 1)
                        add(ex2, k, period, first == b, r["2連単配当"], 5)
                # ---- レース単位：上位3艇（勝率＋展示）の3連複・3連単BOX ----
                score = {b: nat_rank[b] + ex_rank[b] for b in allb}
                top3 = sorted(allb, key=lambda b: score[b])[:3]
                fuku_hit = sorted(map(int, r["3連複"].split("-"))) == sorted(top3)
                for k in [f"全体", f"場:{place}", f"1号艇{cls1}", f"{w}"]:
                    add(box, "3連複 上位3艇（1点）・" + k, period, fuku_hit, r["3連複配当"], 1)
                    add(box, "3連単 上位3艇BOX（6点）・" + k, period, fuku_hit, r["3連単配当"], 6)

    def show(title, table, min_n):
        rows = []
        for k, v in table.items():
            a, b = v["学習"], v["検証"]
            if a[0] < min_n or b[0] < min_n:
                continue
            ra, rb = a[3] / a[2], b[3] / b[2]
            rows.append((min(ra, rb), ra, rb, k, a, b))
        rows.sort(reverse=True)
        print(f"\n## {title}（学習・検証の両方で回収率が高い順）")
        print("| 条件 | 学習 件数 | 学習 的中率 | 学習 回収率 | 検証 件数 | 検証 的中率 | 検証 回収率 |")
        print("|---|---|---|---|---|---|---|")
        for _, ra, rb, k, a, b in rows[:25]:
            print(f"| {k} | {a[0]} | {100 * a[1] / a[0]:.1f}% | {100 * ra:.0f}% | {b[0]} | {100 * b[1] / b[0]:.1f}% | {100 * rb:.0f}% |")

    show("単勝（その艇の1着）", win, 150)
    show("2連単（その艇の1着から5点流し）", ex2, 150)
    show("上位3艇の3連複・3連単BOX", box, 150)


if __name__ == "__main__":
    main(sys.argv[1:])
