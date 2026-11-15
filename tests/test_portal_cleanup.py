"""Operational view can start clean without deleting evidence or blocking new cameras."""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from snatchguard.central.app import create_app
from snatchguard.central.db import Store


def test_archive_is_persistent_reversible_and_new_events_appear(tmp_path):
    store = Store(tmp_path / 'central.db')
    store.insert_event({'event_id': 'old', 'level': 'SUSPECT', 'type': 'snatch'}, '/media/old/snapshot.jpg')
    store.upsert_heartbeat({'node_id': 'old-node', 'cameras': [{'id': 'camera', 'status': 'offline'}]})
    # Explicit old timestamp keeps the ordering deterministic at millisecond resolution.
    with store.lock:
        store.db.execute("UPDATE events SET received_at='2026-01-01T00:00:00+00:00'")
        store.db.execute("UPDATE cameras SET last_seen='2026-01-01T00:00:00+00:00'")
        store.db.execute("UPDATE nodes SET last_seen='2026-01-01T00:00:00+00:00'")
        store.db.commit()
    archived = store.archive_activity()
    assert archived['events'] == 1 and archived['cameras'] == 1
    assert store.list_events() == [] and store.cameras() == [] and store.nodes() == []
    assert store.get_event('old')['snapshot'] == '/media/old/snapshot.jpg'
    assert store.stats('2026-01-01')['open'] == 0
    store.db.close()
    store = Store(tmp_path / 'central.db')
    assert not store.list_events()
    store.insert_event({'event_id': 'new', 'level': 'SUSPECT', 'type': 'snatch'}, None)
    assert [e['id'] for e in store.list_events()] == ['new']
    store.restore_activity()
    assert {e['id'] for e in store.list_events()} == {'old', 'new'}
    store.db.close()


def test_camera_network_excludes_video_jobs_and_stale_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        for node, site in [('live', 'Campus'), ('upload', 'Uploaded footage'),
                           ('demo', 'Real CCTV demonstration · training clip'), ('stale', 'Campus')]:
            app.state.store.upsert_heartbeat({'node_id': node, 'site': site, 'cameras': [{'id': 'c', 'status': 'online'}]})
        with app.state.store.lock:
            old = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
            app.state.store.db.execute('UPDATE cameras SET last_seen=? WHERE node_id=?', (old, 'stale'))
            app.state.store.db.commit()
        assert [c['node_id'] for c in client.get('/api/v1/cameras').json()['cameras']] == ['live']
        page = client.get('/').text
        assert 'start-real-demo' not in page
        assert 'Connect camera or upload video' in page
