#!/usr/bin/env python3
"""過去データで「買い方のルール」を検証する（的中率と回収率）。

1号艇を1着に固定し、2・3着の候補3艇（6点）や4艇（12点）を
いろいろな基準で選んだときの成績を比べる。払戻は実際の3連単配当を使う。

使い方:
  python3 backtest.py <開始YYYYMMDD> <終了YYYYMMDD> <キャッシュ用ディレクトリ>
"""
import itertools
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import stats as S


def load(day, cache):
    ymd, ym = day.strftime("%y%m%d"), day.strftime("%Y%m")
    k = S.lzh_text(S.fetch(f"https://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh", f"{cache}/k{ymd}.lzh"))
    b = S.lzh_text(S.fetch(f"https://www1.mbrace.or.jp/od2/B/{ym}/b{ymd}.lzh", f"{cache}/b{ymd}.lzh"))
    try:
        return (S.parse_k(k) if k else []), (S.parse_b(b) if b else {})
    except Exception:
        return [], {}


def ranks(vals, reverse):
    """{艇: 値} → {艇: 順位(1始まり)}。値が大きいほど上（reverse=True）"""
    order = sorted(vals, key=lambda b: vals[b], reverse=reverse)
    return {b: i + 1 for i, b in enumerate(order)}


def main(argv):
    start, end = (datetime.strptime(x, "%Y%m%d").date() for x in argv[:2])
    cache = argv[2]
    days = [start + timedelta(n) for n in range((end - start).days + 1)]

    # 戦略名 → {グループ: [件数, 的中, 投資, 払戻]}
    res = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

    def add(name, group, picks, r, pts):
        a = res[name][group]
        a[0] += 1
        a[2] += 100 * pts
        if r["3連単"] in picks:
            a[1] += 1
            a[3] += r["3連単配当"]

    with ThreadPoolExecutor(8) as ex:
        for races, prog in ex.map(lambda d: load(d, cache), days):
            for r in races:
                if len(r["艇"]) < 6 or not r["3連単配当"]:
                    continue
                if any(x["進入"] != x["艇"] for x in r["艇"]):
                    continue  # 枠なりのレースだけ（進入の影響を除く）
                boats = prog.get((r["場"], r["R"]), {}).get("艇", {})
                if len(boats) < 6:
                    continue
                ex_t = {x["艇"]: x["展示"] for x in r["艇"] if x["展示"]}
                if len(ex_t) < 6:
                    continue
                others = range(2, 7)
                nat = {b: boats[b]["全国勝率"] for b in others}
                loc = {b: boats[b]["当地勝率"] for b in others}
                mot = {b: boats[b]["モーター2率"] for b in others}
                exh = {b: -ex_t[b] for b in others}  # 速いほど上
                r_nat, r_exh, r_loc, r_mot = (ranks(v, True) for v in (nat, exh, loc, mot))
                # 総合（順位の和）
                combo = {b: r_nat[b] + r_exh[b] for b in others}
                combo3 = {b: r_nat[b] + r_exh[b] + r_loc[b] for b in others}

                cls1 = boats[1]["級別"]
                all_a1 = all(boats[b]["級別"] == "A1" for b in range(1, 7))
                groups = ["全体", f"1号艇{cls1}"] + (["全員A1"] if all_a1 else [])

                def top(score, n, smaller_better=False):
                    return sorted(others, key=lambda b: score[b], reverse=not smaller_better)[:n]

                cand = {
                    "全国勝率上位": top(nat, 3),
                    "当地勝率上位": top(loc, 3),
                    "展示タイム上位": top(exh, 3),
                    "モーター上位": top(mot, 3),
                    "勝率＋展示（順位和）": top(combo, 3, True),
                    "勝率＋展示＋当地": top(combo3, 3, True),
                    "内枠固定(2,3,4)": [2, 3, 4],
                }
                for g in groups:
                    for name, c in cand.items():
                        picks = {f"1-{x}-{y}" for x, y in itertools.permutations(c, 2)}
                        add(f"1-{name}3艇（6点）", g, picks, r, 6)
                    # 4艇に広げる（12点）
                    c4 = top(combo, 4, True)
                    picks = {f"1-{x}-{y}" for x, y in itertools.permutations(c4, 2)}
                    add("1-勝率＋展示4艇（12点）", g, picks, r, 12)
                    # 2着を2艇に絞り、3着は総流し（2着2艇×4 = 8点）
                    c2 = top(combo, 2, True)
                    picks = {f"1-{x}-{y}" for x in c2 for y in others if y != x}
                    add("1-勝率＋展示2艇-全（8点）", g, picks, r, 8)
                    # 2着を1艇に絞り、3着は総流し（4点）
                    c1 = top(combo, 1, True)
                    picks = {f"1-{c1[0]}-{y}" for y in others if y != c1[0]}
                    add("1-勝率＋展示1艇-全（4点）", g, picks, r, 4)
                    # 1号艇の総流し（20点）
                    picks = {f"1-{x}-{y}" for x, y in itertools.permutations(others, 2)}
                    add("1-全-全（20点）", g, picks, r, 20)

    for g in ["全体", "1号艇A1", "1号艇A2", "1号艇B1", "全員A1"]:
        print(f"\n## {g}")
        print("| 買い方 | レース数 | 的中率 | 回収率 |")
        print("|---|---|---|---|")
        rows = []
        for name, d in res.items():
            n, hit, inv, pay = d[g]
            if n:
                rows.append((pay / inv, name, n, hit))
        for roi, name, n, hit in sorted(rows, reverse=True):
            print(f"| {name} | {n} | {100 * hit / n:.1f}% | {100 * roi:.1f}% |")


if __name__ == "__main__":
    main(sys.argv[1:])
