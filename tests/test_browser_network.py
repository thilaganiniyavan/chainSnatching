"""Multiple independent devices, ownership, lifecycle, and cross-camera evidence."""
import asyncio
import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from snatchguard.central.app import create_app
from snatchguard.central.browsercam import BrowserCameras
from snatchguard.central.db import Store
from snatchguard.central.footage import FootageRunner

JPEG_A = b'\xff\xd8' + b'a' * 150
JPEG_B = b'\xff\xd8' + b'b' * 150


@pytest.fixture
def network(tmp_path, monkeypatch):
    def start(self, source, name, port, **kwargs):
        if self.job and not self.stop_requested:
            raise RuntimeError('Already running')
        self.job = {'id': self.directory.name, 'node_id': 'footage-' + self.directory.name,
                    'camera_id': 'camera-01', 'name': name, 'kind': 'live camera'}
        self.sources[self.job['node_id']] = {'kind': 'browsercam'}
        self.stop_requested = False
        return self.status()
    monkeypatch.setattr(FootageRunner, 'start', start)
    monkeypatch.setattr(FootageRunner, 'status', lambda self: {**self.job, 'state': 'stopped' if self.stop_requested else 'running'} if self.job else {'state': 'idle'})
    def stop(self):
        self.stop_requested = True
        return self.status()
    monkeypatch.setattr(FootageRunner, 'stop', stop)
    monkeypatch.setenv('SNATCHGUARD_AI_REVIEW', '0')
    app = create_app(tmp_path, {'key'}, dashboard_password='pw')
    login = {'authorization': 'Basic ' + base64.b64encode(b'operator:pw').decode()}
    with TestClient(app, headers=login) as client:
        yield app, client


def credentials(client):
    session = client.get('/api/v1/browsercam/token').json()
    return session, {'x-camera-id': session['id'], 'x-camera-token': session['token']}


def socket(session):
    return f"/api/v1/browsercam/ws?id={session['id']}&token={session['token']}"


def test_two_devices_independent_frames_detectors_outputs_and_stop(network):
    app, client = network
    a, ah = credentials(client); b, bh = credentials(client)
    with client.websocket_connect(socket(a)) as wa, client.websocket_connect(socket(b)) as wb:
        wa.send_bytes(JPEG_A); assert wa.receive_json()['ready']
        wb.send_bytes(JPEG_B); assert wb.receive_json()['ready']
        ja = client.post('/api/v1/footage/browser-live', headers=ah, json={'name': 'Gate', 'site': 'Campus'}).json()
        jb = client.post('/api/v1/footage/browser-live', headers=bh, json={'name': 'Road', 'site': 'Campus'}).json()
        assert ja['node_id'] != jb['node_id']
        assert len(client.get('/api/v1/cameras').json()['cameras']) == 2
        sa, sb = [app.state.browser_cameras.sessions[x['id']] for x in (a, b)]
        assert sa.jpg == JPEG_A and sb.jpg == JPEG_B
        # Exercise the exact stream generator without an indefinitely buffered HTTP response.
        endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/api/v1/browsercam/stream.mjpg')
        async def first_frame():
            response = endpoint(id=a['id'], key='key')
            frame = await anext(response.body_iterator)
            await response.body_iterator.aclose()
            return frame
        assert JPEG_A in asyncio.run(first_frame())
        assert JPEG_B not in asyncio.run(first_frame())
        for eid, job in [('a', ja), ('b', jb)]:
            app.state.store.insert_event({'event_id': eid, 'type': 'snatch', 'level': 'SUSPECT',
                                         'node_id': job['node_id'], 'camera_id': 'camera-01'}, None)
        assert [x['id'] for x in client.get('/api/v1/footage/output', headers=ah).json()['incidents']] == ['a']
        assert [x['id'] for x in client.get('/api/v1/footage/output', headers=bh).json()['incidents']] == ['b']
        assert len(client.get('/api/v1/events').json()) == 2
        assert client.post('/api/v1/browsercam/stop', headers=ah).status_code == 200
        assert client.get('/api/v1/browsercam/status', headers=bh).json()['state'] == 'running'
        cams = client.get('/api/v1/cameras').json()['cameras']
        assert next(c for c in cams if c['node_id'] == ja['node_id'])['status'] == 'offline'
        hb = {'node_id': ja['node_id'], 'cameras': [{'id': 'camera-01', 'status': 'online'}]}
        assert client.post('/api/v1/heartbeat', json=hb, headers={'x-api-key': 'key'}).json()['ignored']


def test_camera_ownership_permissions_capacity_and_replay(network, monkeypatch):
    app, client = network
    assert client.get('/api/v1/browsercam/token', headers={'authorization': ''}).status_code == 401
    a, ah = credentials(client); b, bh = credentials(client)
    wrong = {**ah, 'x-camera-token': b['token']}
    assert client.post('/api/v1/browsercam/stop', headers=wrong).status_code == 403
    assert client.get('/api/v1/footage/output', headers=wrong).status_code == 403
    assert client.post('/api/v1/footage/browser-live', headers=ah, json={}).status_code == 409
    with client.websocket_connect(socket(a)) as ws:
        ws.send_bytes(JPEG_A); ws.receive_json()
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(socket(a)):
                pass
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(socket(b), headers={'origin': 'https://attacker.example'}):
                pass
    assert app.state.browser_cameras.sessions[a['id']].closed
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(socket(a)):
            pass
    app.state.browser_cameras.limit = 1
    assert client.get('/api/v1/browsercam/token').status_code == 429
    assert client.post('/api/v1/browsercam/stop', headers=bh).status_code == 200
    assert client.get('/api/v1/browsercam/token').status_code == 200


def test_stale_sessions_release_resources_and_buffers_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(FootageRunner, 'stop', lambda self: None)
    registry = BrowserCameras(tmp_path, 'key', limit=1)
    s = registry.create(); s.connected = True
    for _ in range(80):
        s.push(b'x' * 500_000)
    assert s.buffered_bytes <= 32 * 1024 * 1024
    s.at = time.time() - 16
    assert registry.expire() == [s]
    assert s.closed and registry.create().id != s.id


def test_cross_camera_suggestions_and_durable_operator_links(network, tmp_path):
    app, client = network
    def event(eid, node, site='Campus', occurred='2026-11-15T12:00:00+00:00', **kw):
        app.state.store.insert_event({'event_id': eid, 'node_id': node, 'camera_id': 'c', 'site': site,
                                     'type': 'snatch', 'level': 'SUSPECT', 'occurred_at': occurred, **kw}, None)
    event('a', 'one'); event('b', 'two', occurred='2026-11-15T12:01:00+00:00')
    event('same-camera', 'one'); event('far', 'three', occurred='2026-11-15T12:10:00+00:00')
    event('other-site', 'four', site='Elsewhere'); event('generic', 'five', site='Live camera')
    event('hidden', 'six', ai_review={'state': 'rejected'})
    data = client.get('/api/v1/events/a/related').json()
    assert [x['event']['id'] for x in data['candidates']] == ['b']
    assert not client.get('/api/v1/events/generic/related').json()['candidates']
    assert client.post('/api/v1/events/a/links', json={'event_id': 'same-camera'}).status_code == 422
    assert client.post('/api/v1/events/a/links', json={'event_id': 'hidden'}).status_code == 404
    r = client.post('/api/v1/events/a/links', json={'event_id': 'b', 'relation': 'possible_same_person', 'note': 'Review both clips'})
    assert r.status_code == 200 and not r.json()['candidates']
    assert client.get('/api/v1/events/b/related').json()['links'][0]['event']['id'] == 'a'
    reopened = Store(tmp_path / 'central.db')
    assert reopened.event_links('a')[0]['note'] == 'Review both clips'
    reopened.db.close()
    assert client.delete('/api/v1/events/a/links/b').json()['links'] == []


def test_vercel_camera_origin_must_be_explicitly_allowed(tmp_path, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    monkeypatch.setenv('SNATCHGUARD_CAMERA_ORIGINS', 'https://snatchguard.vercel.app')
    app = create_app(tmp_path, {'key'})
    with TestClient(app) as client:
        session, headers = credentials(client)
        with client.websocket_connect(socket(session), headers={'origin': 'https://snatchguard.vercel.app'}) as ws:
            ws.send_bytes(JPEG_A)
            assert ws.receive_json()['ready']
        session, headers = credentials(client)
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(socket(session), headers={'origin': 'https://untrusted.vercel.app'}):
                pass
