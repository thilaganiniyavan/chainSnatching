"""Stage 1: find candidate videos of real chain snatchings and of normal street CCTV.

Runs many search queries (several Indian languages, cities, day/night) through yt-dlp's search
(no API key), keeps the metadata only, and drops obvious non-CCTV content by title: pranks,
films, animation, AI, awareness skits, compilations of unrelated crimes.

    python -m snatchguard.webdata.discover --per-query 60
Output: ~/snatch_web_dataset/meta/candidates.jsonl (one JSON object per unique video id)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATASET = Path.home() / "snatch_web_dataset"

CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Delhi", "Mumbai", "Pune", "Kolkata", "Ahmedabad", "Coimbatore",
          "Madurai", "Mysuru", "Vijayawada", "Visakhapatnam", "Lucknow", "Jaipur", "Ghaziabad", "Noida",
          "Gurugram", "Thane", "Nagpur", "Indore", "Surat", "Kochi", "Thiruvananthapuram", "Ludhiana",
          "Chandigarh", "Bhopal", "Patna", "Kanpur", "Agra", "Warangal", "Tirupati", "Salem", "Trichy"]

SNATCH_QUERIES = [
    "chain snatching cctv footage", "chain snatching caught on camera", "chain snatchers cctv",
    "bike borne chain snatchers cctv", "chain snatching bike cctv video", "gold chain snatched cctv",
    "mangalsutra snatching cctv", "chain snatching woman cctv", "chain snatching night cctv",
    "chain snatching caught on cctv camera india", "snatchers on bike cctv woman chain",
    "chain snatching incident cctv viral", "chain snatching old woman cctv",
    "chain snatching morning walk cctv", "chain snatching temple cctv",
    "chain snatcher caught cctv footage police", "chain snatching scooty cctv",
    "चेन स्नैचिंग सीसीटीवी", "चेन स्नेचिंग सीसीटीवी वीडियो", "गले से चेन छीनी सीसीटीवी",
    "చైన్ స్నాచింగ్ సీసీ కెమెరా", "செயின் பறிப்பு சிசிடிவி", "ಸರಗಳ್ಳತನ ಸಿಸಿಟಿವಿ",
    "चेन चोरी सीसीटीवी", "మెడలో గొలుసు లాక్కెళ్లిన సీసీ కెమెరా", "செயின் பறிப்பு சிசிடிவி காட்சி",
    "mobile snatching cctv bike", "purse snatching cctv bike india", "bag snatching bike cctv india",
] + [f"chain snatching {c} cctv" for c in CITIES]

# Raw uploads (police, home/shop DVR owners, CCTV channels) rather than news packages: news cuts
# the footage short, zooms, and draws circles and arrows over it. Rider-on-pedestrian snatches of
# phones and bags from other countries are the same motion pattern on different cameras.
RAW_SNATCH_QUERIES = [
    "chain snatching cctv footage full video", "chain snatching cctv original video", "chain snatching dvr recording",
    "chain snatching caught on my house cctv", "chain snatching outside house cctv camera",
    "chain snatching cctv camera 1", "chain snatching ip camera footage", "chain snatching cctv no edit",
    "police cctv chain snatching footage", "district police chain snatching cctv",
    "commissionerate police chain snatching cctv", "chain snatching cctv footage released by police",
    "mobile snatching cctv karachi", "mobile snatching cctv lahore", "street crime cctv karachi snatching",
    "snatching cctv footage pakistan bike", "dacoity cctv karachi street bike", "mobile snatching cctv india bike",
    "robo en moto camara de seguridad", "motochorros camara de seguridad", "arrebato moto camara de seguridad",
    "robo de celular en moto camara de seguridad", "motochorro arrebata celular camara",
    "phone snatching moped cctv london", "moped snatch cctv", "bag snatch motorbike cctv",
    "snatch theft cctv malaysia", "ragut cctv motosikal", "kes ragut rakaman cctv", "jambret cctv motor",
    "chain snatching cctv sri lanka", "ছিনতাই সিসিটিভি", "chain snatching cctv bangladesh",
]

# Training pool (2026-10-23): the same rider/pedestrian grab is filmed by street cameras across South-East Asia,
# Latin America and more Indian states. These feed training only; the test sets stay frozen and are de-duplicated
# against everything downloaded here.
INTL_SNATCH_QUERIES = [
    "giật dây chuyền camera", "camera ghi lại cảnh giật dây chuyền", "cướp giật dây chuyền camera an ninh",
    "cướp giật điện thoại camera", "camera nóng cướp giật", "giật túi xách camera an ninh", "cướp giật xe máy camera",
    "กระชากสร้อย กล้องวงจรปิด", "วิ่งราวสร้อย กล้องวงจรปิด", "กระชากกระเป๋า กล้องวงจรปิด", "วิ่งราวโทรศัพท์ กล้องวงจรปิด",
    "ragut rantai leher cctv", "ragut beg tangan cctv", "kes ragut terakam cctv", "peragut motosikal cctv",
    "jambret kalung terekam cctv", "penjambretan terekam cctv", "jambret hp terekam cctv", "begal jambret cctv",
    "arrebato de cadena camara de seguridad", "robo de cadena moto camara", "le arrancan la cadena camara",
    "motoladrones camara de seguridad", "roban celular a mujer moto camara",
    "roubo de corrente moto câmera de segurança", "arrancou cordão câmera de segurança", "assalto moto câmera flagra",
    "飞车抢夺 监控", "飞车抢包 监控视频", "抢项链 监控", "機車搶劫 監視器",
    "sangili parippu cctv", "mala pottikkal cctv", "മാല പൊട്ടിക്കൽ സിസിടിവി", "മാല കവർന്നു സിസിടിവി ദൃശ്യം",
    "चेन स्नॅचिंग सीसीटीव्ही", "चेन स्नेचर सीसीटीवी में कैद", "गले से चेन झपटी सीसीटीवी", "चेन लूट सीसीटीवी फुटेज",
    "ચેન સ્નેચિંગ સીસીટીવી", "চেইন ছিনতাই সিসিটিভি", "ਚੇਨ ਝਪਟ ਸੀਸੀਟੀਵੀ", "ଚେନ ଛଡ଼ାଇ ସିସିଟିଭି",
    "செயின் பறிப்பு சிசிடிவி வீடியோ", "நகை பறிப்பு சிசிடிவி", "చైన్ స్నాచర్ సీసీ టీవీ", "ಚಿನ್ನದ ಸರ ಕಳ್ಳತನ ಸಿಸಿಟಿವಿ",
    "chain snatching cctv compilation", "chain snatchers caught on cctv compilation", "snatch theft cctv compilation",
    "phone snatching cctv compilation", "chain snatching cctv 2024", "chain snatching cctv 2025",
    "chain snatcher arrested cctv footage", "chain snatching woman walking cctv street",
]

NEWS_CHANNEL = re.compile(
    r"news|\btv|(?<!cc)tv\b|samachar|times|express|today|\blive\b|24|channel|media|bharat|aaj ?tak|ndtv|abp|zee|etv|"
    r"polimer|thanthi|\bsun\b|v6|ntv|tv9|tv5|sakshi|abn|mirror|republic|\bani\b|oneindia|otv|ibc|asianet|"
    r"manorama|mathrubhumi|puthiya|jaya|lokmat|loksatta|dainik|patrika|amar ujala|jagran|kesari|jagbani|ptc|"
    r"sandesh|gstv|vtv|doordarshan|wion|hindustan|navbharat|bulletin|khabar|vaartha|eenadu|hmtv|kairali|"
    r"reporter|suvarna|tribune|inkhabar|dinamalar|\btak\b|latestly|brut|associated press|scroll|nyoooz|"
    r"headline|updates|darpan|prime9|kalinga|saam|savera|tiroki|telugubhoomi|samayam|inextlive|exclusive|"
    r"\bfox\d*|cnn|bbc|sky|telemundo|univision|noticias|noticiero|diario|canal|astro awani|berita|kompas|"
    r"geo|dawn|ary|samaa|dunya|express|92|bol\b|hum\b|aaj\b", re.I)


def source_of(item: dict) -> str:
    """'news' for a broadcaster's upload (edited package), else 'raw' (police, owner, CCTV channel)."""
    return "news" if NEWS_CHANNEL.search(item.get("channel") or "") else "raw"


NORMAL_QUERIES = [
    "cctv footage street india", "street cctv camera india live", "traffic cctv footage india",
    "cctv camera road india footage", "market street cctv india", "residential street cctv india",
    "cctv footage colony street", "shop cctv street view india", "night street cctv footage india",
    "cctv footage lane india night", "junction traffic cctv india", "cctv camera footage busy road",
    "bus stop cctv footage india", "temple street cctv footage", "cctv footage footpath india",
    "street view cctv recording full", "cctv dvr footage street outside house india",
    "traffic police cctv junction footage", "night ir cctv street footage", "cctv footage village road india",
] + [f"{c} street cctv footage" for c in CITIES[:15]]

REJECT_TITLE = re.compile(
    r"prank|movie|film|scene|drama|serial|web ?series|trailer|song|animat|cartoon|\bai\b|a\.i\.|generated|"
    r"awareness|skit|comedy|funny|spoof|parody|game|gta|minecraft|roblox|tutorial|how to|reaction|podcast|"
    r"interview|debate|explained|documentary|episode|short film|staged|fake|mock ?drill|re-?enact|"
    r"recreat|simulation|dashcam|dash cam|drone|vlog|unboxing|review|asmr|live stream|livestream", re.I)


def search(query: str, n: int) -> list[dict]:
    cmd = [sys.executable, "-m", "yt_dlp", f"ytsearch{n}:{query}", "--flat-playlist", "--dump-json",
           "--no-warnings", "--ignore-errors"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        return []
    items = []
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        items.append({"id": d.get("id"), "title": d.get("title") or "", "channel": d.get("channel") or
                      d.get("uploader") or "", "duration": d.get("duration"), "url": d.get("url") or
                      f"https://www.youtube.com/watch?v={d.get('id')}", "view_count": d.get("view_count")})
    return items


def keep(item: dict, kind: str) -> str | None:
    """None to keep, else the reason for dropping."""
    dur = item.get("duration")
    if not item.get("id") or dur is None:
        return "no duration (live/unavailable)"
    if REJECT_TITLE.search(item["title"]):
        return "title: not real CCTV"
    if kind == "snatch":
        if dur > 900:
            return "too long for an incident clip"
        if not re.search(r"snatch|chain|स्नैच|स्नेच|स्नॅच|चेन|छीन|झपट|స్నాచ|గొలుసు|பறிப்பு|செயின்|ಸರ|mangalsutra|purse|bag|"
                         r"mobile|phone|robo|arrebat|motochorro|motoladr|ragut|jambret|ছিনতাই|dacoit|street crime|"
                         r"giật|cướp|กระชาก|วิ่งราว|cadena|cord[aã]o|corrente|roubo|assalto|抢|搶|മാല|ચેન|চেইন|ਚੇਨ|ଚେନ|"
                         r"நகை|parippu|pottik", item["title"], re.I):
            return "title: not about a snatching"
    else:
        if dur < 120:
            return "too short for normal footage"
        if re.search(r"snatch|theft|robbery|chor|stolen|accident|fight|attack|murder|crash|fire|caught", item["title"], re.I):
            return "title: an incident, not normal footage"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=60)
    ap.add_argument("--kind", choices=["snatch", "normal", "both"], default="both")
    ap.add_argument("--raw", action="store_true", help="run the raw-upload snatch queries instead")
    ap.add_argument("--intl", action="store_true", help="run the training-pool queries (more countries, languages)")
    args = ap.parse_args()
    out_path = DATASET / "meta" / "candidates.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            seen[d["id"]] = d
    if args.raw:
        jobs = [("snatch", q) for q in RAW_SNATCH_QUERIES]
    elif args.intl:
        jobs = [("snatch", q) for q in INTL_SNATCH_QUERIES]
    else:
        jobs = ([("snatch", q) for q in SNATCH_QUERIES] if args.kind in ("snatch", "both") else []) + \
               ([("normal", q) for q in NORMAL_QUERIES] if args.kind in ("normal", "both") else [])
    dropped: dict[str, int] = {}
    for i, (kind, q) in enumerate(jobs, 1):
        found = search(q, args.per_query)
        new = 0
        for item in found:
            reason = keep(item, kind)
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            if item["id"] in seen:
                seen[item["id"]].setdefault("queries", []).append(q)
                continue
            item.update({"kind": kind, "queries": [q], "source": source_of(item)})
            seen[item["id"]] = item
            new += 1
        print(f"[{i}/{len(jobs)}] {kind:6s} {q[:50]:50s} {len(found):3d} results, {new:3d} new "
              f"(total {sum(1 for v in seen.values() if v['kind'] == 'snatch')} snatch / "
              f"{sum(1 for v in seen.values() if v['kind'] == 'normal')} normal)", flush=True)
        out_path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in seen.values()), encoding="utf-8")
    for v in seen.values():
        v["source"] = source_of(v)
    out_path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in seen.values()), encoding="utf-8")
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
