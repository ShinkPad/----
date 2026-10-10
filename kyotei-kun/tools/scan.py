#!/usr/bin/env python3
"""今から締切のレースをまとめて v2 で判定し、「買い」のレースを探す

- 全国の場の締切時刻を取り、締切まで --from〜--to 分のレースを対象にする
- 展示タイムが出ているレースだけ判定する（出ていないものは「展示待ち」と表示）
- 判定した予想は predictions.jsonl に記録する（同じレースは最初の1回だけ）

使い方: python3 scan.py [--from 3] [--to 40] [--budget 500]
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import official as O
import predict as P
import predict2 as P2

JST = timezone(timedelta(hours=9))
LOG = os.path.join(P2.DATA, "predictions.jsonl")


def deadlines(hd):
    def get(j):
        try:
            page = O.fetch_cached(f"{O.BASE}/race/raceindex?jcd={j:02d}&hd={hd}")
        except Exception:
            return j, []
        return j, re.findall(r">(\d{1,2}:\d{2})<", page)[:12]
    with ThreadPoolExecutor(8) as ex:
        return {j: t for j, t in ex.map(get, range(1, 25)) if len(t) == 12}


def logged(hd):
    if not os.path.exists(LOG):
        return set()
    return {(x["jcd"], x["R"], x["date"]) for x in map(json.loads, open(LOG, encoding="utf-8"))}


def main(argv):
    opt = {"--from": 3, "--to": 40, "--budget": 500}
    for k in opt:
        if k in argv:
            opt[k] = int(argv[argv.index(k) + 1])
    now = datetime.now(JST)
    hd = now.strftime("%Y%m%d")
    P2.load_history(hd, P2.DEFAULT_CACHE)
    races = []
    for j, ts in deadlines(hd).items():
        for r, t in enumerate(ts, 1):
            h, m = map(int, t.split(":"))
            dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
            mins = (dl - now).total_seconds() / 60
            if opt["--from"] <= mins <= opt["--to"]:
                races.append((dl, j, r, t))
    races.sort()
    done = logged(hd)

    def run(x):
        dl, j, r, t = x
        try:
            txt, p, t3 = P2.predict(j, r, hd, opt["--budget"], log=False)
        except Exception as e:
            return x, None, repr(e)
        return x, (txt, p), None

    print(f"# {now:%H:%M} 時点：締切まで{opt['--from']}〜{opt['--to']}分のレース {len(races)}件\n")
    print("| 締切 | 場 | R | 判定 | 本命（1着確率） | 買い目 | 硬いモード |")
    print("|---|---|---|---|---|---|---|")
    buys = []
    with ThreadPoolExecutor(3) as ex:
        for (dl, j, r, t), res, err in ex.map(run, races):
            name = O.PLACES[j - 1]
            if err:
                print(f"| {t} | {name} | {r} | 取得できず | - | - | - |")
                continue
            txt, p = res
            if "展示なし" in txt:
                print(f"| {t} | {name} | {r} | 展示待ち | - | - | - |")
                continue
            v, sv = txt.split("## 判定")[1].split("## 硬いモード")
            go = "✅" in v[:10]
            pat = r"\| (3連単|2連単) \| (\d-\d(?:-\d)?) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \|"
            bets = re.findall(pat, v)
            safe = [(a, b, float(c) / 100, float(d), float(e)) for a, b, c, d, e in re.findall(pat, sv)]
            fav = int(p.argmax()) + 1
            print(f"| {t} | {name} | {r} | {'✅ 買い' if go else '⛔ 見送り'} | {fav}（{100 * p.max():.0f}%） | "
                  f"{' '.join(f'{a}{b}' for a, b, *_ in bets) if go else '-'} | "
                  f"{' '.join(f'{a}{b}' for a, b, *_ in safe) if safe else '-'} |")
            if (j, r, hd) not in done:
                cands = [(a, b, float(c) / 100, float(d), float(e)) for a, b, c, d, e in bets]
                P.log_prediction(j, r, hd, p, cands if go else [], go, safe)
            if go or safe:
                buys.append((t, name, r, txt))
    for t, name, r, txt in buys:
        print(f"\n---\n\n## {t} {name} {r}R\n")
        print(txt.split("## 判定")[0].split("## 2連単")[0])
        print("## 判定" + txt.split("## 判定")[1])


if __name__ == "__main__":
    main(sys.argv[1:])
