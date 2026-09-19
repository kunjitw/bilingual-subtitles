"""依作品分的播放列表：外掛（app/plugins.py）替影片標上「哪部作品、第幾季、第幾集」，這裡把它們排成
作品 → 季 → 集 的樹。沒有標的影片（YouTube 這類）不會出現在這個列表，只在一般的播放列表。

一部影片的標記（存在 media.series_info，JSON）：
  {"series": "作品名", "season": "第一季", "season_order": 1, "episode": 2, "episode_label": "2",
   "version": "配音版"（選填，同一季的不同版本分開列）}
"""
import json
import math

LIMITS = {"series": 120, "season": 40, "episode_label": 20, "version": 20}


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def clean(raw) -> dict | None:
    """外掛給的標記：只留認得的欄位，型別不對的丟掉；沒有作品名就當作沒標。"""
    if not isinstance(raw, dict):
        return None
    out = {}
    for key, limit in LIMITS.items():
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and key == "episode_label":
            v = f"{v:g}"
        if isinstance(v, str) and v.strip():
            out[key] = " ".join(v.split())[:limit]
    if "series" not in out:
        return None
    for key in ("season_order", "episode"):
        n = _num(raw.get(key))
        if n is not None:
            out[key] = n
    if "episode_label" not in out and "episode" in out:
        out["episode_label"] = f"{out['episode']:g}"
    return out


def dumps(info: dict | None) -> str | None:
    return json.dumps(info, ensure_ascii=False, sort_keys=True) if info else None


def loads(raw) -> dict | None:
    if not raw:
        return None
    try:
        return clean(json.loads(raw) if isinstance(raw, str) else raw)
    except ValueError:
        return None


def _inf(v):
    return v if v is not None else math.inf


def tree(media: list[dict], subtitled: set | None = None) -> list[dict]:
    """media：db.list_media() 的列（要有 id、created_at、series_info）。subtitled：已經有字幕的影片 id。
    回傳 [{"key", "title", "count", "subtitled", "seasons": [{"key", "title", "count", "subtitled", "items": [id...]}]}]；
    作品照最近加入的排前面，季照 season_order（沒有的排後面）、同一季原版在前，集照 episode（沒有的照加入時間）。"""
    subtitled = subtitled or set()
    works: dict[str, dict] = {}
    for m in media:
        info = loads(m.get("series_info"))
        if not info:
            continue
        w = works.setdefault(info["series"], {"title": info["series"], "latest": 0.0, "seasons": {}})
        w["latest"] = max(w["latest"], float(m.get("created_at") or 0))
        season, version = info.get("season", ""), info.get("version", "")
        s = w["seasons"].setdefault((season, version), {"season": season, "version": version,
                                                        "order": info.get("season_order"), "items": []})
        if s["order"] is None:
            s["order"] = info.get("season_order")
        s["items"].append((_inf(info.get("episode")), info.get("episode_label", ""),
                           float(m.get("created_at") or 0), m["id"]))
    out = []
    for w in sorted(works.values(), key=lambda w: (-w["latest"], w["title"])):
        seasons = []
        for s in sorted(w["seasons"].values(), key=lambda s: (_inf(s["order"]), s["season"], s["version"] != "",
                                                              s["version"])):
            ids = [i[3] for i in sorted(s["items"])]
            title = " ".join(x for x in (s["season"], s["version"]) if x) or "其他"
            seasons.append({"key": f"{w['title']}␟{s['season']}␟{s['version']}", "title": title,
                            "count": len(ids), "subtitled": sum(1 for i in ids if i in subtitled), "items": ids})
        out.append({"key": w["title"], "title": w["title"], "count": sum(s["count"] for s in seasons),
                    "subtitled": sum(s["subtitled"] for s in seasons), "seasons": seasons})
    return out
