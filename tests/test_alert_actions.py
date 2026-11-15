"""Video removal is reversible; detection sounds are triggered before review completes."""
import json

from fastapi.testclient import TestClient

from snatchguard.central.app import Broadcaster, create_app


def test_remove_restore_and_late_clip_do_not_remove_alert(tmp_path, monkeypatch):
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        store = app.state.store
        store.insert_event({'event_id': 'a', 'type': 'snatch', 'level': 'SUSPECT'}, None)
        store.set_clip('a', '/media/a/clip.mp4')
        response = client.delete('/api/v1/events/a/video')
        assert response.status_code == 200
        assert response.json()['clip'] is None
        assert response.json()['payload']['clip_removed']
        assert client.get('/api/v1/events').json() == []
        assert client.get('/api/v1/events?deleted=true').json()[0]['id'] == 'a'
        # A retry / completed trim cannot unexpectedly bring a removed video back.
        store.set_clip('a', '/media/a/trimmed.mp4')
        assert client.get('/api/v1/events/a').json()['clip'] is None
        restored = client.post('/api/v1/events/a/video/restore').json()
        assert restored['clip'] == '/media/a/trimmed.mp4'
        assert not restored['payload'].get('clip_removed')
        assert client.delete('/api/v1/events/missing/video').status_code == 404


def test_immediate_detection_signal_emitted_once_even_while_review_pending(tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(Broadcaster, 'publish', lambda self, kind, data: messages.append((kind, data)))
    app = create_app(tmp_path, {'key'}, reviewer=object())
    with TestClient(app) as client:
        event = {'event_id': 'detected', 'type': 'snatch', 'level': 'SUSPECT', 'node_id': 'live'}
        for _ in range(2):
            assert client.post('/api/v1/events', data={'meta': json.dumps(event)}, headers={'x-api-key': 'key'}).status_code == 200
        detections = [data for kind, data in messages if kind == 'detection']
        assert len(detections) == 1 and detections[0]['id'] == 'detected'
        assert not [data for kind, data in messages if kind == 'event']
        assert client.get('/api/v1/events').json() == []


def test_tamper_trash_hides_alert_and_updates_counts_without_losing_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        store = app.state.store
        store.insert_event({'event_id': 'tamper', 'type': 'tamper', 'level': 'TAMPER'}, '/media/tamper/snapshot.jpg')
        store.insert_event({'event_id': 'incident', 'type': 'snatch', 'level': 'SUSPECT'}, None)
        assert client.get('/api/v1/stats').json()['tamper'] == 1
        response = client.delete('/api/v1/events/tamper')
        assert response.status_code == 200 and response.json()['payload']['deleted']
        assert [e['id'] for e in client.get('/api/v1/events').json()] == ['incident']
        assert client.get('/api/v1/stats').json()['tamper'] == 0
        assert store.get_event('tamper')['snapshot'] == '/media/tamper/snapshot.jpg'
        assert client.delete('/api/v1/events/incident').status_code == 200
        assert client.delete('/api/v1/events/missing').status_code == 404


def test_legacy_video_removal_survives_polling_and_can_be_restored(tmp_path, monkeypatch):
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        store = app.state.store
        store.insert_event({'event_id': 'legacy', 'type': 'snatch', 'level': 'SUSPECT'}, None)
        store.set_clip('legacy', '/media/legacy/clip.mp4')
        client.post('/api/v1/events/legacy/review', json={'status': 'new', 'note': '[Video removed from alert] Original note'})
        for _ in range(2):
            event = client.get('/api/v1/events/legacy').json()
            assert event['clip'] is None and event['payload']['clip_removed']
        restored = client.post('/api/v1/events/legacy/video/restore').json()
        assert restored['clip'] == '/media/legacy/clip.mp4'
        assert 'Original note' in restored['review_note']


def test_deleted_filter_moves_whole_alert_and_restores_it(tmp_path, monkeypatch):
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        store = app.state.store
        store.insert_event({'event_id': 'video', 'type': 'snatch', 'level': 'SUSPECT'}, None)
        store.set_clip('video', '/media/video/clip.mp4')
        assert client.delete('/api/v1/events/video').status_code == 200
        assert client.get('/api/v1/events').json() == []
        assert client.get('/api/v1/events?deleted=true').json()[0]['clip'] == '/media/video/clip.mp4'
        assert len(client.get('/api/v1/events?include_deleted=true').json()) == 1
        assert client.get('/api/v1/stats').json()['open'] == 0
        assert client.post('/api/v1/events/video/restore').status_code == 200
        assert client.get('/api/v1/events?deleted=true').json() == []
        assert client.get('/api/v1/events').json()[0]['id'] == 'video'
        assert client.get('/api/v1/stats').json()['open'] == 1
