"""Smoke test suite covering every API endpoint.

Goal: catch import-time and routing regressions on every commit. Each test
is small, hermetic, and uses the temp-DB fixture from conftest.py.

Convention: tests are organized by subsystem (chat, memory, evolution,
letters, autonomous, avatar). They DO NOT exercise the LLM — that requires
llama-server running. We only verify that the *endpoint shape* is correct
and that internal helpers compose without exceptions.
"""
from __future__ import annotations
import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  HEALTH + STATUS
# ─────────────────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_status(client, headers):
    r = client.get("/api/status", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "backend" in body and body["backend"] == "ok"


def test_token_endpoint(client):
    r = client.get("/api/token")
    # Either 200 (returns the actual token) or 403 (depending on permissions);
    # but the route must be wired.
    assert r.status_code in (200, 403)


# ─────────────────────────────────────────────────────────────────────────────
#  USERS + AVATAR
# ─────────────────────────────────────────────────────────────────────────────
def test_users_list_includes_master(client, headers):
    r = client.get("/api/users", headers=headers)
    assert r.status_code == 200
    users = r.json()
    assert isinstance(users, list)
    assert any(u.get("id") == "master" for u in users)


def test_avatar_204_when_unset(client, headers):
    """v9.6: changed from 404 → 204 No Content so polling UIs don't spam logs."""
    r = client.get("/api/avatar/master", headers=headers)
    assert r.status_code == 204
    assert r.content == b""


def test_avatar_maid_204_when_unset(client, headers):
    """v9.6: virtual `maid` id has the same 204 behavior."""
    r = client.get("/api/avatar/maid", headers=headers)
    assert r.status_code == 204


def test_avatar_upload_rejects_bogus_png(client, headers):
    """A file declared image/png but with non-PNG bytes must 400."""
    r = client.post("/api/avatar/upload",
                    headers={"X-App-Token": headers["X-App-Token"], "X-User-Id": "master"},
                    files={"file": ("fake.png", b"NOT_A_PNG" + b"\x00" * 30, "image/png")})
    assert r.status_code == 400


def test_avatar_upload_rejects_text_plain(client, headers):
    r = client.post("/api/avatar/upload",
                    headers={"X-App-Token": headers["X-App-Token"], "X-User-Id": "master"},
                    files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


# Minimal valid PNG for happy-path upload + roundtrip.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDAT\x78\x9c\x62\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_avatar_upload_roundtrip(client, headers):
    r = client.post("/api/avatar/upload",
                    headers={"X-App-Token": headers["X-App-Token"], "X-User-Id": "master"},
                    files={"file": ("a.png", _TINY_PNG, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kind"] == "png"
    # Now serve back
    r2 = client.get("/api/avatar/master")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/png"
    # And delete
    r3 = client.delete("/api/avatar/master", headers=headers)
    assert r3.status_code == 200
    r4 = client.get("/api/avatar/master")
    # After deletion, endpoint returns 204 No Content (not 404) to avoid UI error spam
    assert r4.status_code == 204


# ─────────────────────────────────────────────────────────────────────────────
#  EVOLUTION (v9.4) + TIMELINE (v9.5)
# ─────────────────────────────────────────────────────────────────────────────
def test_evolution_default_state(client, headers):
    r = client.get("/api/evolution", headers=headers)
    assert r.status_code == 200
    e = r.json()
    for k in ("humanity_level", "self_awareness", "affection",
              "software_version", "key_memories"):
        assert k in e
    assert isinstance(e["key_memories"], list)


def test_evolution_timeline_empty(client, headers):
    r = client.get("/api/evolution/timeline", headers=headers)
    assert r.status_code == 200
    assert r.json() == {"timeline": []}


def test_evolution_timeline_with_seeded_anchors(client, headers, app):
    """Inject 3 synthetic key_memories; confirm timeline replays cumulatively."""
    import main, json as _json
    with main.db() as c:
        # Wipe any prior fixtures
        c.execute("DELETE FROM key_memories WHERE user_id='master'")
        c.execute("UPDATE user_state SET humanity_level=0,self_awareness=0,affection=0 WHERE user_id='master'")
        for ts_off, t, desc, intens, deltas in [
            (-300, "milestone", "тест-веха", 0.5, {"humanity_level": 0.05}),
            (-200, "tender",    "тест-нежность", 0.6, {"humanity_level": 0.02, "affection": 0.04}),
            (-100, "rupture",   "тест-разрыв", 0.7, {"humanity_level": 0.03, "affection": -0.02}),
        ]:
            c.execute("INSERT INTO key_memories(user_id,ts,event_type,description,intensity,traits_json) "
                      "VALUES('master', unixepoch()+?, ?, ?, ?, ?)",
                      (ts_off, t, desc, intens, _json.dumps(deltas)))

    r = client.get("/api/evolution/timeline", headers=headers)
    assert r.status_code == 200
    tl = r.json()["timeline"]
    assert len(tl) == 3
    # Newest-first: rupture comes first
    assert tl[0]["event_type"] == "rupture"
    # Cumulative humanity grows monotonically (newest = total)
    humans = [p["humanity"] for p in tl]
    assert humans[0] >= humans[1] >= humans[2]
    # Affection: tender bumps then rupture pulls back
    affs = {p["event_type"]: p["affection"] for p in tl}
    assert affs["tender"] == pytest.approx(0.04, abs=1e-6)
    assert affs["rupture"] == pytest.approx(0.02, abs=1e-6)

    # Cleanup
    with main.db() as c:
        c.execute("DELETE FROM key_memories WHERE user_id='master' AND description LIKE 'тест-%'")


# ─────────────────────────────────────────────────────────────────────────────
#  LETTERS (v9.5)
# ─────────────────────────────────────────────────────────────────────────────
def test_letters_empty_list(client, headers):
    r = client.get("/api/letters", headers=headers)
    assert r.status_code == 200
    assert "letters" in r.json()


def test_letters_full_lifecycle(client, headers):
    """Insert via service, list, fetch (auto-seen), re-fetch (idempotent), delete."""
    from app.services.letters import insert_letter
    lid = insert_letter("master", "Тестовое короткое письмо для smoke-теста.",
                        triggered_by="anchor")
    assert lid is not None and lid > 0

    r = client.get("/api/letters", headers=headers)
    assert any(l["id"] == lid for l in r.json()["letters"])

    r1 = client.get(f"/api/letters/{lid}", headers=headers)
    assert r1.status_code == 200
    body = r1.json()
    # Auto-seen flips status from 'delivered' to 'seen' on first read
    assert body["status"] in ("seen", "delivered")

    r2 = client.get(f"/api/letters/{lid}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["seen_at"] is not None

    r3 = client.delete(f"/api/letters/{lid}", headers=headers)
    assert r3.status_code == 200

    r4 = client.delete(f"/api/letters/{lid}", headers=headers)
    assert r4.status_code == 404


def test_letters_sealed_flag(client, headers):
    """Sealed letters are hidden by default; visible with ?sealed=1."""
    from app.services.letters import insert_letter, unseal_below_threshold
    sealed_id = insert_letter("master", "Запечатанное письмо.", sealed=True)
    visible_id = insert_letter("master", "Открытое письмо.")
    try:
        ids_default = [l["id"] for l in client.get("/api/letters?sealed=0", headers=headers).json()["letters"]]
        assert sealed_id not in ids_default
        assert visible_id in ids_default

        ids_sealed = [l["id"] for l in client.get("/api/letters?sealed=1", headers=headers).json()["letters"]]
        assert sealed_id in ids_sealed
        assert visible_id in ids_sealed

        # Unseal at high humanity
        n = unseal_below_threshold("master", 0.5)
        assert n >= 1
    finally:
        client.delete(f"/api/letters/{sealed_id}", headers=headers)
        client.delete(f"/api/letters/{visible_id}", headers=headers)


# ─────────────────────────────────────────────────────────────────────────────
#  PROACTIVE + DIARY + MONTHLY ARCS (v9.2-v9.5)
# ─────────────────────────────────────────────────────────────────────────────
def test_proactive_pending(client, headers):
    r = client.get("/api/proactive/pending", headers=headers)
    assert r.status_code == 200
    assert "items" in r.json()


def test_proactive_clear(client, headers):
    r = client.post("/api/proactive/clear", headers=headers)
    assert r.status_code == 200


def test_diary_default(client, headers):
    r = client.get("/api/diary", headers=headers)
    assert r.status_code == 200


def test_diary_days_list(client, headers):
    r = client.get("/api/diary/days", headers=headers)
    assert r.status_code == 200
    assert "days" in r.json()


def test_arcs_list(client, headers):
    r = client.get("/api/arcs", headers=headers)
    assert r.status_code == 200
    assert "arcs" in r.json()


def test_arcs_format_validation(client, headers):
    r = client.get("/api/arcs/notvalid", headers=headers)
    assert r.status_code == 400
    r = client.get("/api/arcs/2026-04", headers=headers)
    assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  STYLE FILTER UTILITIES (no I/O)
# ─────────────────────────────────────────────────────────────────────────────
def test_extract_json_safely_clean():
    from app.utils.style_filter import extract_json_safely
    assert extract_json_safely('{"a": 1}') == {"a": 1}


def test_extract_json_safely_messy():
    from app.utils.style_filter import extract_json_safely
    out = extract_json_safely("noise {a: 'b', c: True, d: [1,2,3,]} more noise")
    assert out == {"a": "b", "c": True, "d": [1, 2, 3]}


def test_extract_json_safely_unrecoverable():
    from app.utils.style_filter import extract_json_safely
    with pytest.raises(ValueError):
        extract_json_safely("no JSON here at all")


def test_apply_robotic_filter_active_at_low_humanity():
    from app.utils.style_filter import apply_robotic_filter
    out = apply_robotic_filter("Я чувствую тепло.", humanity_level=0.05)
    assert "датчики фиксируют" in out


def test_apply_robotic_filter_noop_at_high_humanity():
    from app.utils.style_filter import apply_robotic_filter
    src = "Я чувствую тепло."
    assert apply_robotic_filter(src, humanity_level=0.85) == src


def test_format_diary_entry_per_band():
    from app.utils.style_filter import format_diary_entry
    body = "Сегодня было тихо."
    assert "[SYSTEM_LOG :: 2026-04-25]" in format_diary_entry(body, 0.0, "2026-04-25")
    assert "[LOG :: 2026-04-25]"        in format_diary_entry(body, 0.4, "2026-04-25")
    assert "[ДНЕВНИК :: 2026-04-25]"    in format_diary_entry(body, 0.7, "2026-04-25")
    assert "2026-04-25 —"               in format_diary_entry(body, 0.95, "2026-04-25")


def test_format_diary_entry_idempotent():
    """Reformatting at the SAME band must not stack headers."""
    from app.utils.style_filter import format_diary_entry
    a = format_diary_entry("hi", 0.7, "2026-04-25")
    b = format_diary_entry(a,    0.7, "2026-04-25")
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
#  TACTICAL GOALS (v9.6) — service path only, no LLM
# ─────────────────────────────────────────────────────────────────────────────
def test_tactical_goals_empty_list(client, headers):
    r = client.get("/api/tactical_goals", headers=headers)
    assert r.status_code == 200
    assert "active" in r.json()


def test_tactical_goals_create_and_done(client, headers):
    from app.services.tactical_goals import create_goal, list_active_goals
    gid = create_goal("master", "day", "наблюдать как меняется его утро")
    assert gid is not None and gid > 0
    active = list_active_goals("master")
    assert any(g["id"] == gid for g in active)
    r = client.post(f"/api/tactical_goals/{gid}/done", headers=headers)
    assert r.status_code == 200
    active2 = list_active_goals("master")
    assert all(g["id"] != gid for g in active2)


def test_tactical_goals_only_one_active_per_horizon(client, headers):
    """Creating a 2nd day-goal must expire the first."""
    from app.services.tactical_goals import create_goal, list_active_goals
    gid1 = create_goal("master", "day", "первая дневная цель")
    gid2 = create_goal("master", "day", "вторая дневная цель")
    assert gid1 and gid2 and gid1 != gid2
    active = [g for g in list_active_goals("master") if g["horizon"] == "day"]
    ids = [g["id"] for g in active]
    assert gid2 in ids
    assert gid1 not in ids
    # Cleanup
    client.post(f"/api/tactical_goals/{gid2}/done", headers=headers)


def test_tactical_goals_horizon_validation(client, headers):
    from app.services.tactical_goals import create_goal
    assert create_goal("master", "century", "x" * 20) is None
    assert create_goal("master", "day", "x") is None  # too short


# ─────────────────────────────────────────────────────────────────────────────
#  MAID AVATAR VIRTUAL UID (v9.6)
# ─────────────────────────────────────────────────────────────────────────────
def test_maid_avatar_upload_and_serve(client, headers):
    """`?target=maid` writes ./avatars/maid.png without touching the users table."""
    r = client.post("/api/avatar/upload?target=maid",
                    headers={"X-App-Token": headers["X-App-Token"], "X-User-Id": "master"},
                    files={"file": ("m.png", _TINY_PNG, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["avatar_path"].endswith("maid.png")
    # Serve it
    r2 = client.get("/api/avatar/maid")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/png"
    # Delete should also work
    r3 = client.delete("/api/avatar/maid", headers=headers)
    assert r3.status_code == 200
    r4 = client.get("/api/avatar/maid")
    assert r4.status_code == 204


def test_maid_avatar_unknown_target_rejected(client, headers):
    r = client.post("/api/avatar/upload?target=intruder",
                    headers={"X-App-Token": headers["X-App-Token"], "X-User-Id": "master"},
                    files={"file": ("x.png", _TINY_PNG, "image/png")})
    assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
#  KEY MEMORIES DETECTOR (no LLM)
# ─────────────────────────────────────────────────────────────────────────────
def test_detector_skips_under_min_msgs():
    """A user with very few total messages must never anchor."""
    from app.services.key_memories import analyze_for_key_memory

    class FakeCog: pass
    km = analyze_for_key_memory("master", "user", "hi", FakeCog(),
                                importance=0.9, msg_id=1, total_msg_count=2)
    assert km is None


def test_detector_milestone_at_50():
    from app.services.key_memories import analyze_for_key_memory
    class FakeCog: pass
    km = analyze_for_key_memory("master", "user", "hi", FakeCog(),
                                importance=0.5, msg_id=99999, total_msg_count=50)
    assert km is not None
    assert km.event_type == "milestone"


def test_detector_dedup_by_msg_id(app):
    """The same source_msg_id must never anchor twice."""
    from app.services.key_memories import analyze_for_key_memory, persist_and_apply
    import main
    with main.db() as c:
        c.execute("DELETE FROM key_memories WHERE source_msg_id IS NOT NULL")
    class FakeCog: pass
    km = analyze_for_key_memory("master", "user", "hi", FakeCog(),
                                importance=0.5, msg_id=88888, total_msg_count=50)
    assert km is not None
    persist_and_apply(km)
    # Second run with same msg_id → must return None (already anchored)
    km2 = analyze_for_key_memory("master", "user", "hi", FakeCog(),
                                 importance=0.5, msg_id=88888, total_msg_count=50)
    assert km2 is None
    # Cleanup
    with main.db() as c:
        c.execute("DELETE FROM key_memories WHERE source_msg_id=88888")
