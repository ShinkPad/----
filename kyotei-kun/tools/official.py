#!/usr/bin/env python3
"""BOATRACE公式サイトから、1レース分の予想材料をまとめて取得して表にする。

取得するもの:
  - 出走表   : 級別、全国・当地の勝率/2連率/3連率、平均ST、F/L、モーター・ボートの2連率/3連率、今節成績
  - 直前情報 : 展示タイム、チルト、部品交換、調整重量、スタート展示、気象
  - オッズ   : 3連単（全120通り）。人気順の上位を表示
  - 選手ページ: コース別の進入率、1着率・2着率・3着率、平均ST
  - 当日結果 : その場の、このレースより前の全レースの着順・進入・決まり手・3連単配当と、その日の傾向

使い方:
  python3 official.py <場番号 1-24> <レース番号 1-12> [YYYYMMDD] [--odds-top N] [--json]
  例) python3 official.py 20 7 20261008

標準ライブラリだけで動く。
"""
import html
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

BASE = "https://www.boatrace.jp/owpc/pc"
PLACES = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
          "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as res:
        return res.read().decode("utf-8")


def tokens(fragment):
    parts = (html.unescape(re.sub(r"\s+", " ", p)).strip() for p in re.split(r"<[^>]+>", fragment))
    return [p for p in parts if p]


def to_half(s):
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def parse_racelist(page):
    title = re.search(r'<h3 class="title16_titleDetail__add2020">(.*?)</h3>', page, re.S)
    times = tokens(page[page.find("締切予定時刻"):page.find("締切予定時刻") + 1500])[1:13]
    boats = []
    for body in re.findall(r'<tbody class="\s*is-fs12\s*">(.*?)</tbody>', page, re.S):
        t = tokens(body)
        reg, cls = t[1].rstrip(" /"), t[2]
        b = {"枠": int(to_half(t[0])), "登番": reg, "級別": cls, "選手": t[3], "支部/出身": t[4], "年齢/体重": t[5],
             "F": t[6], "L": t[7], "平均ST": t[8],
             "全国勝率": t[9], "全国2連率": t[10], "全国3連率": t[11],
             "当地勝率": t[12], "当地2連率": t[13], "当地3連率": t[14],
             "モーター": t[15], "モーター2連率": t[16], "モーター3連率": t[17],
             "ボート": t[18], "ボート2連率": t[19], "ボート3連率": t[20]}
        rest = t[21:]
        sts = [x for x in rest if re.fullmatch(r"[FL]?\.\d\d|[FL]\d?", x)]
        n = len(sts)
        first_st = rest.index(sts[0]) if sts else len(rest)
        head = rest[:first_st]
        k = len(head) // 2
        finishes = [to_half(x) for x in rest[first_st + n:]]
        b["今節"] = [{"R": head[i], "進入": head[k + i] if k + i < len(head) else "-",
                      "ST": sts[i] if i < n else "-", "着": finishes[i] if i < len(finishes) else "-"}
                     for i in range(k)]
        boats.append(b)
    return {"タイトル": tokens(title.group(1)) if title else [], "締切時刻": times, "艇": boats}


def parse_beforeinfo(page):
    bodies = re.findall(r'<tbody class="is-fs12\s*">(.*?)</tbody>', page, re.S)
    boats = []
    for body in bodies:
        t = tokens(body)
        b = {"枠": int(to_half(t[0])), "選手": t[1], "体重": t[2], "展示": t[3], "チルト": t[4]}
        r = t.index("R") if "R" in t else len(t)
        b["プロペラ/部品交換"] = " ".join(t[5:r]) or "-"
        after = t[t.index("進入") + 1:] if "進入" in t else []
        adj = [x for x in after if re.fullmatch(r"\d+\.\d", x)]
        b["調整重量"] = adj[0] if adj else "-"
        boats.append(b)
    start = []
    for m in re.finditer(r'table1_boatImage1Number is-type\d">\s*(\d)\s*<.*?table1_boatImage1Time[^>]*>\s*([^<]*)<',
                         page, re.S):
        start.append({"枠": int(m.group(1)), "ST": m.group(2).strip()})
    i = page.find("weather1_body")
    wseg = page[i:page.find("weather1_stand", i)] if i >= 0 else ""
    pairs = re.findall(r'LabelTitle">([^<]*)</span>\s*(?:<span class="weather1_bodyUnitLabelData">([^<]*)</span>)?', wseg)
    w = {k.strip(): v.strip() for k, v in pairs}
    sky = [k.strip() for k, v in pairs if not v]
    wd = re.search(r'is-windDirection">\s*<p class="weather1_bodyUnitImage is-wind(\d+)"', wseg)
    when = re.search(r'weather1_title[^>]*>\s*([^<]*?)\s*<', page)
    weather = {"時点": when.group(1) if when else "-", "気温": w.get("気温", "-"), "天気": sky[0] if sky else "-",
               "風速": w.get("風速", "-"), "風向コード": wd.group(1) if wd else "-", "水温": w.get("水温", "-"),
               "波高": w.get("波高", "-")}
    return {"艇": boats, "スタート展示": start, "気象": weather}


def parse_odds3t(page):
    odds = {}
    tb = re.search(r'<tbody class="is-p3-0">(.*?)</tbody>', page, re.S)
    if not tb:
        return odds
    second = [None] * 6
    for tr in re.findall(r"<tr>(.*?)</tr>", tb.group(1), re.S):
        cells = re.findall(r"<td([^>]*)>(.*?)</td>", tr, re.S)
        col, i = 0, 0
        while i < len(cells) and col < 6:
            attrs, val = cells[i]
            if "rowspan" in attrs:
                second[col] = re.sub(r"<[^>]+>", "", val).strip()
                i += 1
                attrs, val = cells[i]
            third = re.sub(r"<[^>]+>", "", val).strip()
            o = re.sub(r"<[^>]+>", "", cells[i + 1][1]).strip()
            odds[f"{col + 1}-{second[col]}-{third}"] = o
            i += 2
            col += 1
    return odds


def parse_course(page):
    heads = ["コース別進入率", "コース別3連対率", "コース別平均スタートタイミング", "コース別スタート順", "集計期間内"]

    def section(name):
        i = page.find(name)
        if i < 0:
            return ""
        ends = [page.find(h, i + len(name)) for h in heads]
        ends = [e for e in ends if e > i]
        return page[i:min(ends) if ends else len(page)]

    def by_course(sec):
        vals = {}
        for m in re.finditer(r'is-boatColor(\d)[^>]*>\d</th>(.*?)</tr>', sec, re.S):
            t = tokens(m.group(2))
            vals[m.group(1)] = t[0] if t else "-"
        return vals

    res = {}
    for m in re.finditer(r'is-boatColor(\d)[^>]*>\d</th>(.*?)</tr>', section("コース別3連対率"), re.S):
        widths = re.findall(r'class="is-progress" style="width:\s*([\d.]+)%', m.group(2))
        if len(widths) == 3:
            a, b, c = (float(x) for x in widths)
            res[m.group(1)] = {"1着率": a, "2着率": b, "3着率": c, "3連対率": round(a + b + c, 1)}
    return {"進入率": by_course(section("コース別進入率")), "着率": res,
            "平均ST": by_course(section("コース別平均スタートタイミング"))}


def parse_result(page):
    """レース結果ページから、着順・進入（コース順の枠）・ST・決まり手・3連単を取る。"""
    r = {"着順": [], "進入": [], "ST": {}, "決まり手": "-", "3連単": "-", "3連単配当": "-", "3連単人気": "-"}
    for tb in re.findall(r"<table[^>]*>(.*?)</table>", page, re.S):
        t = tokens(tb)
        if t[:2] == ["着", "枠"]:
            body = t[4:]
            for i, x in enumerate(body):
                if re.fullmatch(r"[１２３４５６]", x) and i + 1 < len(body):
                    r["着順"].append(int(body[i + 1]))
        elif t and t[0] == "スタート情報":
            body = t[1:]
            for i in range(0, len(body) - 1, 2):
                waku, st = body[i], body[i + 1]
                r["進入"].append(int(waku))
                r["ST"][int(waku)] = st.split()[0]
        elif t and t[0] == "決まり手" and len(t) > 1:
            r["決まり手"] = t[1]
        elif t[:2] == ["勝式", "組番"] and "3連単" in t:
            i = t.index("3連単")
            r["3連単"] = "".join(t[i + 1:i + 6])
            r["3連単配当"] = t[i + 6]
            r["3連単人気"] = t[i + 7] if i + 7 < len(t) else "-"
    return r


def day_summary(results):
    done = [x for x in results if x["着順"]]
    n = len(done)
    if not n:
        return {"レース数": 0}
    win_course = []
    for x in done:
        w = x["着順"][0]
        win_course.append(x["進入"].index(w) + 1 if w in x["進入"] else None)
    kimarite = {}
    for x in done:
        kimarite[x["決まり手"]] = kimarite.get(x["決まり手"], 0) + 1
    pays = [int(re.sub(r"[^\d]", "", x["3連単配当"])) for x in done if re.search(r"\d", x["3連単配当"])]
    # コース別の1着・2着・3着の回数（進入コースで数える）
    by_course = {c: [0, 0, 0] for c in range(1, 7)}
    for x in done:
        for pos, b in enumerate(x["着順"][:3]):
            if b in x["進入"]:
                by_course[x["進入"].index(b) + 1][pos] += 1
    # 直近3レースの1着コース
    recent = [c for c in win_course[-3:]]
    alerts = []
    for c in range(2, 7):
        if by_course[c][0] >= 2:
            alerts.append(f"{c}コースの1着が{by_course[c][0]}回（{n}レース中）→ {c}号艇の1着を必ず押さえる")
    if n >= 4 and by_course[1][0] / n < 0.4:
        alerts.append(f"1コースの1着が{by_course[1][0]}/{n}回と少ない → 1号艇の1着固定は避ける（2連複・表裏も検討）")
    if n >= 4 and by_course[1][0] / n >= 0.7:
        alerts.append(f"1コースの1着が{by_course[1][0]}/{n}回と多い → 1号艇の1着を信頼してよい")
    hot3 = [c for c in range(1, 7) if sum(by_course[c]) >= max(3, n * 0.6)]
    if hot3:
        alerts.append("3着以内によく来るコース：" + "・".join(f"{c}コース（{sum(by_course[c])}回）" for c in hot3))
    return {"レース数": n, "1コース1着": win_course.count(1), "コース別着数": by_course,
            "直近の1着コース": recent, "流れアラート": alerts,
            "枠なり以外": sum(1 for x in done if x["進入"] and x["進入"] != sorted(x["進入"])),
            "決まり手": kimarite, "万舟": sum(1 for p in pays if p >= 10000),
            "3連単平均配当": round(sum(pays) / len(pays)) if pays else "-"}


def collect(jcd, rno, hd, odds_top=15):
    q = f"rno={rno}&jcd={jcd:02d}&hd={hd}"
    with ThreadPoolExecutor(4) as ex:
        f_list = ex.submit(fetch, f"{BASE}/race/racelist?{q}")
        f_before = ex.submit(fetch, f"{BASE}/race/beforeinfo?{q}")
        f_odds = ex.submit(fetch, f"{BASE}/race/odds3t?{q}")
        racelist = parse_racelist(f_list.result())
        courses = {b["登番"]: ex.submit(fetch, f"{BASE}/data/racersearch/course?toban={b['登番']}")
                   for b in racelist["艇"]}
        before = parse_beforeinfo(f_before.result())
        odds = parse_odds3t(f_odds.result())
        course = {k: parse_course(v.result()) for k, v in courses.items()}
        earlier = [ex.submit(fetch, f"{BASE}/race/raceresult?rno={r}&jcd={jcd:02d}&hd={hd}") for r in range(1, rno)]
        results = [dict(parse_result(f.result()), R=i + 1) for i, f in enumerate(earlier)]
    ranked = sorted(((k, float(v)) for k, v in odds.items() if re.fullmatch(r"[\d.]+", v)), key=lambda x: x[1])
    return {"場": PLACES[jcd - 1], "R": rno, "日付": hd, "出走表": racelist, "直前情報": before,
            "コース別成績": course, "3連単オッズ": odds, "人気上位": ranked[:odds_top],
            "当日結果": results, "当日傾向": day_summary(results)}


def render(d):
    out = []
    rl, bf = d["出走表"], d["直前情報"]
    deadline = rl["締切時刻"][d["R"] - 1] if len(rl["締切時刻"]) >= d["R"] else "-"
    out.append(f"# {d['場']} {d['R']}R（{d['日付']}、締切 {deadline}）")
    out.append(" ".join(rl["タイトル"]))
    out.append("\n## 出走表")
    out.append("| 枠 | 選手 | 級 | F/L | 平均ST | 全国 勝率/2連 | 当地 勝率/2連 | モーター No/2連 | ボート No/2連 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for b in rl["艇"]:
        out.append(f"| {b['枠']} | {b['選手']} | {b['級別']} | {b['F']}/{b['L']} | {b['平均ST']} | "
                   f"{b['全国勝率']}/{b['全国2連率']} | {b['当地勝率']}/{b['当地2連率']} | "
                   f"{b['モーター']}/{b['モーター2連率']} | {b['ボート']}/{b['ボート2連率']} |")
    out.append("\n## 今節成績（R:進入→着順 ST）")
    for b in rl["艇"]:
        s = "、".join(f"{x['R']}:{x['進入']}→{x['着']}({x['ST']})" for x in b["今節"]) or "-"
        out.append(f"- {b['枠']} {b['選手']}：{s}")
    out.append("\n## 枠のコースでの成績（選手ページ、直近の集計）")
    out.append("| 枠 | 選手 | このコースの進入率 | 1着率 | 2着率 | 3着率 | 3連対率 | コース別平均ST |")
    out.append("|---|---|---|---|---|---|---|---|")
    for b in rl["艇"]:
        c, k = d["コース別成績"].get(b["登番"], {}), str(b["枠"])
        r = c.get("着率", {}).get(k, {})
        out.append(f"| {k} | {b['選手']} | {c.get('進入率', {}).get(k, '-')} | {r.get('1着率', '-')}% | "
                   f"{r.get('2着率', '-')}% | {r.get('3着率', '-')}% | {r.get('3連対率', '-')}% | "
                   f"{c.get('平均ST', {}).get(k, '-')} |")
    out.append("\n## 直前情報")
    out.append("| 枠 | 選手 | 体重 | 調整重量 | 展示 | チルト | プロペラ/部品交換 | スタート展示ST |")
    out.append("|---|---|---|---|---|---|---|---|")
    st = {s["枠"]: s["ST"] for s in bf["スタート展示"]}
    for b in bf["艇"]:
        out.append(f"| {b['枠']} | {b['選手']} | {b['体重']} | {b['調整重量']} | {b['展示'] or '-'} | {b['チルト']} | "
                   f"{b['プロペラ/部品交換']} | {st.get(b['枠'], '-')} |")
    if bf["スタート展示"]:
        out.append("スタート展示の並び（内から）：" + " ".join(str(s["枠"]) for s in bf["スタート展示"]))
    w = bf["気象"]
    out.append(f"\n気象（{w['時点']}）：気温{w['気温']} {w['天気']} 風速{w['風速']}（風向コード{w['風向コード']}） "
               f"水温{w['水温']} 波高{w['波高']}")
    out.append("\n## 当日のこれまでのレース（この場）")
    sm = d["当日傾向"]
    if sm.get("レース数"):
        km = "、".join(f"{k} {v}" for k, v in sm["決まり手"].items())
        out.append(f"- {sm['レース数']}レース中、1コース1着 **{sm['1コース1着']}回**、枠なり以外の進入 {sm['枠なり以外']}回、"
                   f"万舟 {sm['万舟']}回、3連単平均配当 {sm['3連単平均配当']}円")
        out.append(f"- 決まり手：{km}")
        out.append("- コース別の着数（1着/2着/3着）：" + "、".join(
            f"{c}C {v[0]}/{v[1]}/{v[2]}" for c, v in sm["コース別着数"].items()))
        out.append(f"- 直近3レースの1着コース：{sm['直近の1着コース']}")
        if sm["流れアラート"]:
            out.append("\n### ⚠️ 当日の流れアラート（予想に必ず反映する）")
            out.extend(f"- {a}" for a in sm["流れアラート"])
        out.append("\n| R | 着順（枠） | 進入（内から枠） | 決まり手 | 3連単 | 配当（人気） |")
        out.append("|---|---|---|---|---|---|")
        for x in d["当日結果"]:
            if x["着順"]:
                out.append(f"| {x['R']} | {'-'.join(map(str, x['着順'][:3]))} | {''.join(map(str, x['進入']))} | "
                           f"{x['決まり手']} | {x['3連単']} | {x['3連単配当']}（{x['3連単人気']}） |")
    else:
        out.append("（このレースより前の結果はまだありません）")
    out.append("\n## 3連単オッズ 人気上位")
    out.append(" / ".join(f"{k} {v}" for k, v in d["人気上位"]) or "（まだ発売されていないか、取得できませんでした）")
    return "\n".join(out)


def main(argv):
    top = 15
    if "--odds-top" in argv:
        i = argv.index("--odds-top")
        top = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 1
    jcd, rno = int(args[0]), int(args[1])
    hd = args[2] if len(args) > 2 else date.today().strftime("%Y%m%d")
    d = collect(jcd, rno, hd, top)
    print(json.dumps(d, ensure_ascii=False, indent=1) if "--json" in argv else render(d))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
