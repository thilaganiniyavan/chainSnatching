"""Web discovery title filter: the second training-pool sweep's languages must survive it, skits must not."""

import pytest

from snatchguard.webdata.discover import GLOBAL_SNATCH_QUERIES, keep


def _item(title: str, duration: float = 60.0) -> dict:
    return {"id": "x", "title": title, "duration": duration}


@pytest.mark.parametrize("title", [
    "Ladrão de moto rouba celular de mulher na calçada, câmera flagra",
    "Raponazo en Medellín quedó grabado en cámara de seguridad",
    "Snatcher nahuli sa CCTV sa Quezon City",
    "موبائل چھیننے کی واردات سی سی ٹی وی میں ریکارڈ",
    "خطف موبايل من سيدة كاميرا مراقبة",
    "Kapkaç anı güvenlik kamerasında",
    "Scippo in centro, le telecamere riprendono tutto",
    "Vol à l'arraché filmé par la vidéosurveillance",
    "ひったくり 防犯カメラが捉えた瞬間",
    "오토바이 날치기 CCTV 영상",
])
def test_new_languages_are_kept(title):
    assert keep(_item(title), "snatch") is None


@pytest.mark.parametrize("title", [
    "Phone snatching prank on the street",
    "Raponazo - short film",
    "Ladrão de celular: cena da novela",
])
def test_skits_and_films_are_dropped(title):
    assert keep(_item(title), "snatch") is not None


def test_overlong_videos_are_dropped():
    assert keep(_item("Raponazo cámara de seguridad", duration=1200), "snatch") is not None


def test_every_global_query_title_would_pass_its_own_filter():
    # a query whose own words fail the title filter would only ever find titles the filter throws away
    failing = [q for q in GLOBAL_SNATCH_QUERIES if keep(_item(q), "snatch") is not None]
    assert failing == []


# review kits: rows filled by reviewers on other computers -> training decisions in source-video seconds
from snatchguard.webdata.trainpool import parse_kit_rows  # noqa: E402


def _row(**kw):
    base = {"n": "1", "clip": "clips/0001.mp4", "decision": "", "start_s": "", "end_s": "", "lighting": "",
            "notes": "", "key": "abc#2", "offset_s": "40.00", "duration_s": "30.00"}
    base.update(kw)
    return base


def test_kit_snatch_times_become_source_seconds():
    got, bad = parse_kit_rows([_row(decision="Snatch_Bike", start_s="12,5", end_s="13.5", lighting="Night")])
    assert bad == []
    assert got[0]["decision"] == "snatch_bike" and got[0]["lighting"] == "night"
    assert (got[0]["event_start_s"], got[0]["event_end_s"]) == ("52.50", "53.50")


def test_kit_blank_rows_are_unreviewed_and_non_snatch_needs_no_times():
    got, bad = parse_kit_rows([_row(), _row(n="2", decision="no_event")])
    assert bad == [] and [d["decision"] for d in got] == ["no_event"]


def test_kit_problems_are_reported():
    _, bad = parse_kit_rows([_row(decision="snatch_foot", lighting="day"),                       # no time
                             _row(n="2", decision="snatch_foot", start_s="45", lighting="day"),  # outside clip
                             _row(n="3", decision="snatch_bike", start_s="3"),                   # no lighting
                             _row(n="4", decision="maybe")])
    assert len(bad) == 4
