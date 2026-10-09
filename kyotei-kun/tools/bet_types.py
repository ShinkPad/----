#!/usr/bin/env python3
"""3連単と2連単の買い方を、同じルールの2・3着候補で比べる（過去データ・実際の配当）。

使い方: python3 bet_types.py <開始YYYYMMDD> <終了YYYYMMDD> <キャッシュ用ディレクトリ>
"""
import itertools
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import backtest as B


def main(argv):
    start, end = (datetime.strptime(x, "%Y%m%d").date() for x in argv[:2])
    cache = argv[2]
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    res = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

    def add(name, g, hit, pay, pts):
        a = res[name][g]
        a[0] += 1
        a[2] += 100 * pts
        if hit:
            a[1] += 1
            a[3] += pay

    with ThreadPoolExecutor(8) as ex:
        for races, prog in ex.map(lambda d: B.load(d, cache), days):
            for r in races:
                if len(r["艇"]) < 6 or not r["3連単配当"] or not r.get("2連単配当"):
                    continue
                if any(x["進入"] != x["艇"] for x in r["艇"]):
                    continue
                bo = prog.get((r["場"], r["R"]), {}).get("艇", {})
                ext = {x["艇"]: x["展示"] for x in r["艇"] if x["展示"]}
                if len(bo) < 6 or len(ext) < 6:
                    continue
                o = range(2, 7)
                rn = B.ranks({b: bo[b]["全国勝率"] for b in o}, True)
                re_ = B.ranks({b: -ext[b] for b in o}, True)
                c = sorted(o, key=lambda b: rn[b] + re_[b])
                cls1 = bo[1]["級別"]
                all_a1 = all(bo[b]["級別"] == "A1" for b in range(1, 7))
                # 1号艇の「強さ」: A1で、全国勝率が6艇中1位
                strong1 = cls1 == "A1" and max(bo[b]["全国勝率"] for b in range(1, 7)) == bo[1]["全国勝率"]
                groups = ["全体", f"1号艇{cls1}"] + (["全員A1"] if all_a1 else []) + (["1号艇A1かつ勝率1位"] if strong1 else [])
                t3, t2, p3, p2 = r["3連単"], r["2連単"], r["3連単配当"], r["2連単配当"]
                for g in groups:
                    add("3連単 1-上位3艇-上位3艇（6点）", g, t3 in {f"1-{x}-{y}" for x, y in itertools.permutations(c[:3], 2)}, p3, 6)
                    add("3連単 1-上位2艇-上位3艇（4点）", g, t3 in {f"1-{x}-{y}" for x in c[:2] for y in c[:3] if y != x}, p3, 4)
                    add("2連単 1-上位1艇（1点）", g, t2 == f"1-{c[0]}", p2, 1)
                    add("2連単 1-上位2艇（2点）", g, t2 in {f"1-{x}" for x in c[:2]}, p2, 2)
                    add("2連単 1-上位3艇（3点）", g, t2 in {f"1-{x}" for x in c[:3]}, p2, 3)
                    add("2連単 1-全（5点）", g, t2.startswith("1-"), p2, 5)
                    add("2連単 1-上位2艇＋裏（4点）", g, t2 in {f"1-{x}" for x in c[:2]} | {f"{x}-1" for x in c[:2]}, p2, 4)

    for g in ["全体", "1号艇A1", "1号艇A1かつ勝率1位", "1号艇A2", "1号艇B1", "全員A1"]:
        print(f"\n## {g}")
        print("| 買い方 | レース数 | 的中率 | 回収率 |")
        print("|---|---|---|---|")
        rows = sorted(((d[g][3] / d[g][2], k, d[g][0], d[g][1]) for k, d in res.items() if d[g][0]), reverse=True)
        for roi, k, n, h in rows:
            print(f"| {k} | {n} | {100 * h / n:.1f}% | {100 * roi:.1f}% |")


if __name__ == "__main__":
    main(sys.argv[1:])
