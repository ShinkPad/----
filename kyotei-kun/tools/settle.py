#!/usr/bin/env python3
"""predictions.jsonl の予想を公式結果と照合し、「買い」判定と「見送り」判定の成績を集計する。

使い方: python3 settle.py
- 買い判定のレース：推奨の買い目を各100円で買った場合の回収率
- 見送り判定のレース：もし買っていたら（同じ基準の上位の目を買ったら）どうだったか
"""
import json
import os

import official as O

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "..", "data", "predictions.jsonl")
seen, rows = set(), []
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    key = (r["jcd"], r["R"], r["date"])
    if key in seen:
        rows = [x for x in rows if (x["jcd"], x["R"], x["date"]) != key]
    seen.add(key)
    rows.append(r)  # 同じレースは最後の予想を使う

agg = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}
for r in rows:
    res = O.parse_result(O.fetch(f"{O.BASE}/race/raceresult?rno={r['R']}&jcd={r['jcd']:02d}&hd={r['date']}"))
    if res["3連単"] == "-":
        continue
    t3 = res["3連単"]
    pay3 = int("".join(c for c in res["3連単配当"] if c.isdigit()) or 0)
    a = agg[r["go"]]
    a[0] += 1
    hit = False
    for b in r["bets"]:
        a[2] += 100
        if b["type"] == "3連単" and b["key"] == t3:
            a[3] += pay3
            hit = True
        if b["type"] == "2連単" and b["key"] == t3[:3]:
            a[3] += round(b["odds"] * 100)  # 2連単は記録時のオッズで概算
            hit = True
    a[1] += hit
    print(f"{O.PLACES[r['jcd'] - 1]} {r['R']}R {r['date']} 判定:{'買い' if r['go'] else '見送り'} 結果:{t3} 的中:{'○' if hit else '×'}")
for go, (n, h, inv, pay) in agg.items():
    if n:
        print(f"\n【{'買い' if go else '見送り'}判定】{n}レース 的中{h} 回収率 {100 * pay / inv if inv else 0:.0f}%（見送りは「買っていたら」の数字）")
