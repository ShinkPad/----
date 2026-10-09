#!/usr/bin/env python3
"""BOATRACE公式の過去データ（成績K・番組表B・月間スケジュール）を集計して、
グレード別・レース種別・条件別の傾向を出す。

使い方:
  python3 stats.py <開始YYYYMMDD> <終了YYYYMMDD> [キャッシュ用ディレクトリ] [--json 出力先]
  例) python3 stats.py 20251001 20260930 /tmp/kyotei-cache --json stats.json

必要: pip install lhafile
"""
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import lhafile

PLACES = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
          "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]
Z2H = str.maketrans("０１２３４５６７８９：", "0123456789:")


def fetch(url, path=None):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                data = res.read()
            break
        except Exception:
            data = None
    if data is None:
        return None
    if path:
        with open(path, "wb") as f:
            f.write(data)
    return data


def lzh_text(data):
    if not data or not data.startswith(b"!") and b"-lh" not in data[:16]:
        return None
    tmp = f"/tmp/_k_{os.getpid()}_{id(data)}.lzh"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        lf = lhafile.LhaFile(tmp)
        return lf.read(lf.namelist()[0]).decode("cp932", errors="replace")
    except Exception:
        return None
    finally:
        os.remove(tmp)


# ---------- 月間スケジュール（グレード） ----------
def norm(t):
    return re.sub(r"[\s　]", "", t or "")


def lookup_grade(sched, jcd, hd, title):
    """大会名で照合する（日付の対応はずれることがあるため）。見つからなければ日付で引く。"""
    nt = norm(title)
    best = None
    for st, g in sched.get(("title", jcd), []):
        if nt and st and (st.startswith(nt[:12]) or nt.startswith(st[:12])):
            best = g
    if best:
        return best
    return sched.get((jcd, hd), {}).get("グレード", "不明")
def parse_schedule(page, ym):
    """(場番号, 'YYYYMMDD') -> {'グレード':..., '大会名':...}"""
    y, m = int(ym[:4]), int(ym[4:])
    out = {}
    for jcd, row in re.findall(r"stadium\?jcd=(\d\d)>.*?</th>(.*?)</tr>", page, re.S):
        day = 1
        for attrs, inner in re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S):
            span = re.search(r'colspan\s*="?(\d+)', attrs)
            n = int(span.group(1)) if span else 1
            g = re.search(r"is-gradeColor(\w+)", attrs)
            title = re.sub(r"<[^>]+>", "", inner).strip()
            if g and title:
                out.setdefault(("title", int(jcd)), []).append((norm(title), g.group(1)))
            if g:
                for k in range(n):
                    try:
                        d = date(y, m, day + k)
                    except ValueError:
                        continue
                    out[(int(jcd), d.strftime("%Y%m%d"))] = {"グレード": g.group(1), "大会名": title}
            day += n
    return out


# ---------- 番組表 B ----------
def parse_b(text):
    """{(場番号, R): {'締切': 'HH:MM', '艇': {枠: {'級別','全国勝率','モーター2率'}}}}"""
    out = {}
    for block in re.split(r"\n(?=\d\dBBGN)", text):
        m = re.match(r"(\d\d)BBGN", block)
        if not m:
            continue
        jcd = int(m.group(1))
        for rm in re.finditer(r"　?([０-９]+)Ｒ\s+(\S+).*?締切予定([０-９：]+)(.*?)(?=\n\s*\n|\Z)", block, re.S):
            rno = int(rm.group(1).translate(Z2H))
            boats = {}
            for line in rm.group(4).splitlines():
                bm = re.match(r"([1-6]) (\d{4})(.{4})(\d\d)(.{2})(\d\d)(A1|A2|B1|B2)\s*(\d+\.\d\d)\s*(\d+\.\d\d)"
                              r"\s*(\d+\.\d\d)\s*(\d+\.\d\d)\s*(\d{1,3}?)\s*(\d{1,3}\.\d\d)", line)
                if bm:
                    try:
                        boats[int(bm.group(1))] = {"級別": bm.group(7), "全国勝率": float(bm.group(8)),
                                                   "当地勝率": float(bm.group(10)), "モーター2率": float(bm.group(13))}
                    except ValueError:
                        pass
            out[(jcd, rno)] = {"締切": rm.group(3).translate(Z2H), "艇": boats}
    return out


# ---------- 成績 K ----------
KIMARITE = ["逃げ", "差し", "まくり差し", "まくり", "抜き", "恵まれ"]


def parse_k(text):
    races = []
    for block in re.split(r"\n(?=\d\dKBGN)", text):
        m = re.match(r"(\d\d)KBGN", block)
        if not m:
            continue
        jcd = int(m.group(1))
        tm = re.search(r"＊＊＊　競走成績　＊＊＊\s*\n\s*\n\s*(\S.*?)\s*\n", block)
        title = tm.group(1).strip() if tm else ""
        parts = re.split(r"\n\s+(?=\d{1,2}R\s{2,})", block)
        for p in parts[1:]:
            hm = re.match(r"(\d{1,2})R\s+(\S+)\s.*?H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm", p)
            if not hm:
                continue
            first_line = p.splitlines()[1] if len(p.splitlines()) > 1 else ""
            km = next((k for k in sorted(KIMARITE, key=len, reverse=True) if k in first_line), None)
            rows = []
            for line in p.splitlines():
                rm = re.match(r"\s+(\S\S)\s+([1-6])\s+(\d{4})\s.{8,12}?\s+(\d+)\s+(\d+)\s+([\d.]+|\s*)\s+([1-6]|\s)\s+([FL]?[\d.]+|[KLFS]\d?|\.)?", line)
                if rm:
                    rows.append({"着": rm.group(1), "艇": int(rm.group(2)), "展示": float(rm.group(6)) if rm.group(6).strip() else None,
                                 "進入": int(rm.group(7)) if rm.group(7).strip() else None,
                                 "ST": rm.group(8) or ""})
            pay = re.search(r"３連単\s+(\d-\d-\d)\s+(\d+)\s+人気\s+(\d+)", p)
            pay2 = re.search(r"２連単\s+(\d-\d)\s+(\d+)\s+人気\s+(\d+)", p)
            payf = re.search(r"３連複\s+(\d-\d-\d)\s+(\d+)", p)
            payw = re.search(r"単勝\s+(\d)\s+(\d+)", p)
            races.append({"場": jcd, "大会名": title, "R": int(hm.group(1)), "種別": hm.group(2), "天候": hm.group(4),
                          "風向": hm.group(5), "風速": int(hm.group(6)), "波高": int(hm.group(7)), "決まり手": km,
                          "艇": rows, "3連単": pay.group(1) if pay else None, "3連単配当": int(pay.group(2)) if pay else None,
                          "3連単人気": int(pay.group(3)) if pay else None,
                          "2連単人気": int(pay2.group(3)) if pay2 else None,
                          "2連単": pay2.group(1) if pay2 else None, "2連単配当": int(pay2.group(2)) if pay2 else None,
                          "3連複": payf.group(1) if payf else None, "3連複配当": int(payf.group(2)) if payf else None,
                          "単勝": int(payw.group(1)) if payw else None, "単勝配当": int(payw.group(2)) if payw else None})
    return races


# ---------- 分類 ----------
def race_category(name):
    for k, v in [("優勝戦", "優勝戦"), ("準優", "準優勝戦"), ("ドリーム", "ドリーム戦"), ("特選", "特選・選抜"),
                 ("選抜", "特選・選抜"), ("特賞", "特選・選抜"), ("Ａ戦", "特選・選抜"), ("予選", "予選"), ("一般", "一般戦")]:
        if k in name:
            return v
    return "その他（場独自の企画名）"


GRADE_NAMES = {"SG": "SG", "G1": "G1", "G2": "G2", "G3": "G3", "Ippan": "一般",
               "Lady": "G3オールレディース", "Venus": "ヴィーナスシリーズ", "Rookie": "ルーキーシリーズ", "Takumi": "マスターズリーグ"}


def event_type(title, grade):
    g = GRADE_NAMES.get(grade, grade)
    if g in ("一般", "G3") and re.search(r"ヴィーナス|レディース|女子", title):
        return "女子戦（一般・G3）"
    return g


def grade_group(et):
    if et in ("SG", "G1", "G2"):
        return "SG・G1・G2"
    if et in ("G3オールレディース", "ヴィーナスシリーズ", "女子戦（一般・G3）"):
        return "女子戦"
    if et in ("ルーキーシリーズ", "マスターズリーグ", "G3", "一般"):
        return et
    return "その他"


INDICATORS = {"展示タイム": ("展示", min), "全国勝率": ("全国勝率", max), "当地勝率": ("当地勝率", max),
              "モーター2率": ("モーター2率", max)}


def indicator_hits(r, boats, ind_stats, group):
    """各指標で1位の艇（同率は除外）が、1着・3着以内になった割合を数える。全艇版と2〜6号艇版。"""
    finish = {x["艇"]: x["着"] for x in r["艇"]}
    vals = {}
    for x in r["艇"]:
        v = dict(boats.get(x["艇"], {}))
        v["展示"] = x["展示"]
        vals[x["艇"]] = v
    for name, (key, fn) in INDICATORS.items():
        for scope, cand in (("全艇", range(1, 7)), ("2〜6号艇", range(2, 7))):
            xs = [(vals[b].get(key), b) for b in cand if b in vals and vals[b].get(key) is not None]
            if len(xs) < len(cand):
                continue
            best = fn(v for v, _ in xs)
            tops = [b for v, b in xs if v == best]
            if len(tops) != 1:
                continue
            b = tops[0]
            a = ind_stats[(group, scope, name)]
            a["n"] += 1
            a["win"] += finish.get(b) == "01"
            a["top3"] += finish.get(b) in ("01", "02", "03")
            a["is1"] += b == 1


def time_slot(t):
    if not t:
        return "不明"
    h = int(t.split(":")[0])
    if h < 12:
        return "モーニング（〜12時）"
    if h < 17:
        return "デイ（12〜17時）"
    if h < 20:
        return "ナイター前半（17〜20時）"
    return "ナイター後半・ミッドナイト（20時〜）"


def bucket_wind(v):
    return "0〜1m" if v <= 1 else "2〜3m" if v <= 3 else "4〜5m" if v <= 5 else "6m以上"


def bucket_wave(v):
    return "0〜2cm" if v <= 2 else "3〜5cm" if v <= 5 else "6cm以上"


# ---------- 集計 ----------
class Agg:
    def __init__(self):
        self.d = defaultdict(lambda: defaultdict(float))

    def add(self, key, r):
        a = self.d[key]
        a["n"] += 1
        top = [x for x in r["艇"] if x["着"] in ("01", "02", "03")]
        top.sort(key=lambda x: x["着"])
        win = top[0] if top else None
        if win and win["進入"]:
            a[f"c{win['進入']}"] += 1
        c1 = next((x for x in r["艇"] if x["進入"] == 1), None)
        if c1 and c1["着"] in ("01", "02", "03"):
            a["c1_top3"] += 1
        if r["決まり手"]:
            a["k_" + r["決まり手"]] += 1
        if r["3連単配当"]:
            a["pay_n"] += 1
            a["pay_sum"] += r["3連単配当"]
            a["man"] += r["3連単配当"] >= 10000
            a["low"] += r["3連単配当"] < 1000
            a["fav1"] += r["3連単人気"] == 1
            a["pop_le10"] += (r["3連単人気"] or 999) <= 10
        if r.get("枠なり") is not None:
            a["wakunari"] += r["枠なり"]

    def table(self):
        out = {}
        for k, a in self.d.items():
            n = a["n"]
            if n == 0:
                continue
            row = {"レース数": int(n)}
            for c in range(1, 7):
                row[f"{c}コース1着率"] = round(100 * a[f"c{c}"] / n, 1)
            row["1コース3連対率"] = round(100 * a["c1_top3"] / n, 1)
            for km in KIMARITE:
                row[km] = round(100 * a["k_" + km] / n, 1)
            pn = a["pay_n"] or 1
            row["3連単平均配当"] = round(a["pay_sum"] / pn)
            row["万舟率"] = round(100 * a["man"] / pn, 1)
            row["千円未満率"] = round(100 * a["low"] / pn, 1)
            row["1番人気的中率"] = round(100 * a["fav1"] / pn, 1)
            row["10番人気以内率"] = round(100 * a["pop_le10"] / pn, 1)
            row["枠なり率"] = round(100 * a["wakunari"] / n, 1)
            out[k] = row
        return out


def main(argv):
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        out_json = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    start, end = (datetime.strptime(x, "%Y%m%d").date() for x in argv[:2])
    cache = argv[2] if len(argv) > 2 else "./kyotei-cache"
    os.makedirs(cache, exist_ok=True)
    days = [start + timedelta(n) for n in range((end - start).days + 1)]

    months = sorted({d.strftime("%Y%m") for d in days})
    sched = {}
    for ym in months:
        page = fetch(f"https://www.boatrace.jp/owpc/pc/race/monthlyschedule?ym={ym}", f"{cache}/sched_{ym}.html")
        for k, v in parse_schedule(page.decode("utf-8"), ym).items():
            if k[0] == "title":
                sched.setdefault(k, []).extend(v)
            else:
                sched[k] = v

    def load(d):
        ymd, ym = d.strftime("%y%m%d"), d.strftime("%Y%m")
        k = lzh_text(fetch(f"https://www1.mbrace.or.jp/od2/K/{ym}/k{ymd}.lzh", f"{cache}/k{ymd}.lzh"))
        b = lzh_text(fetch(f"https://www1.mbrace.or.jp/od2/B/{ym}/b{ymd}.lzh", f"{cache}/b{ymd}.lzh"))
        try:
            races = parse_k(k) if k else []
        except Exception as e:
            print(f"K解析エラー {d}: {e}", file=sys.stderr)
            races = []
        try:
            prog = parse_b(b) if b else {}
        except Exception as e:
            print(f"B解析エラー {d}: {e}", file=sys.stderr)
            prog = {}
        return d, races, prog

    groups = {name: Agg() for name in
              ["全体", "グレード", "大会種別", "レース種別", "グレード×レース種別", "1号艇の級別", "1号艇級別×相手最上位",
               "時間帯", "風速", "波高", "1号艇の展示順位", "1号艇のスタートST", "1号艇の全国勝率", "1号艇のモーター2率", "場"]}
    total = 0
    ind_stats = defaultdict(lambda: defaultdict(float))
    with ThreadPoolExecutor(8) as ex:
        for d, races, prog in ex.map(load, days):
            hd = d.strftime("%Y%m%d")
            for r in races:
                if len(r["艇"]) < 6 or not any(x["着"] == "01" for x in r["艇"]):
                    continue
                total += 1
                grade = lookup_grade(sched, r["場"], hd, r["大会名"])
                et = event_type(r["大会名"], grade)
                cat = race_category(r["種別"])
                p = prog.get((r["場"], r["R"]), {})
                boats = p.get("艇", {})
                entries = {x["艇"]: x["進入"] for x in r["艇"]}
                r["枠なり"] = all(entries.get(i) == i for i in range(1, 7))
                b1 = boats.get(1, {})
                cls1 = b1.get("級別", "不明")
                others = [boats[i]["級別"] for i in range(2, 7) if i in boats]
                best_other = min(others) if others else "不明"
                ex_times = sorted((x["展示"], x["艇"]) for x in r["艇"] if x["展示"])
                ex_rank = next((i + 1 for i, (_, b) in enumerate(ex_times) if b == 1), None)
                st1 = next((x["ST"] for x in r["艇"] if x["艇"] == 1), "")
                stv = float(st1) if re.fullmatch(r"\d?\.\d+", st1 or "") else None
                rate1 = b1.get("全国勝率")
                mot1 = b1.get("モーター2率")
                keys = {
                    "全体": "全体", "グレード": GRADE_NAMES.get(grade, grade), "大会種別": et, "レース種別": cat, "グレード×レース種別": f"{et}・{cat}",
                    "1号艇の級別": cls1, "1号艇級別×相手最上位": f"1号艇{cls1}・相手最上位{best_other}",
                    "時間帯": time_slot(p.get("締切")), "風速": bucket_wind(r["風速"]), "波高": bucket_wave(r["波高"]),
                    "1号艇の展示順位": f"{ex_rank}位" if ex_rank else "不明",
                    "1号艇のスタートST": ("〜.10" if stv < .11 else ".11〜.15" if stv < .16 else ".16〜.20" if stv < .21 else ".21〜")
                    if stv is not None else "不明（F・L等）",
                    "1号艇の全国勝率": ("7.00〜" if rate1 >= 7 else "6.00〜6.99" if rate1 >= 6 else "5.00〜5.99" if rate1 >= 5
                                     else "4.00〜4.99" if rate1 >= 4 else "〜3.99") if rate1 is not None else "不明",
                    "1号艇のモーター2率": ("45%〜" if mot1 >= 45 else "38〜45%" if mot1 >= 38 else "30〜38%" if mot1 >= 30
                                      else "〜30%") if mot1 is not None else "不明",
                    "場": PLACES[r["場"] - 1],
                }
                gg = grade_group(et)
                keys["グレード群×レース種別"] = f"{gg}・{cat}"
                keys["グレード群×1号艇の級別"] = f"{gg}・1号艇{cls1}"
                keys["グレード群×1号艇の展示順位"] = f"{gg}・展示{ex_rank}位" if ex_rank else f"{gg}・不明"
                keys["グレード群×風速"] = f"{gg}・{bucket_wind(r['風速'])}"
                for g, k in keys.items():
                    groups.setdefault(g, Agg()).add(k, r)
                indicator_hits(r, boats, ind_stats, gg)
                indicator_hits(r, boats, ind_stats, "全体")
    ind = {}
    for (g, scope, name), a in sorted(ind_stats.items()):
        n = a["n"]
        ind.setdefault(g, {}).setdefault(scope, {})[name] = {
            "レース数": int(n), "1着率": round(100 * a["win"] / n, 1), "3連対率": round(100 * a["top3"] / n, 1),
            "その艇が1号艇の割合": round(100 * a["is1"] / n, 1)}
    result = {"期間": f"{start}〜{end}", "総レース数": total, **{g: a.table() for g, a in groups.items()},
              "指標トップ艇の成績": ind}
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({"総レース数": total, "全体": result["全体"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
