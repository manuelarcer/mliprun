"""PLAN D5: X-Mliprun header on mutations, Host allow-list, WebSocket Origin."""
import pytest
from ase.build import bulk


def test_post_without_header_is_403(client, runs_dir):
    r = client.post("/api/uploads", files=[("files", ("h.xyz", b"1\n\nH 0 0 0\n"))])
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "forbidden"
    assert not [p for p in (runs_dir / "_uploads").iterdir()]
    r = client.post("/api/jobs", json={"calc_type": "optimize"})
    assert r.status_code == 403
    r = client.post("/api/jobs", json={}, headers={"X-Mliprun": "0"})
    assert r.status_code == 403
    r = client.post("/api/jobs/20260101-000000-md-abcdef/cancel")
    assert r.status_code == 403
    assert "access-control-allow-origin" not in r.headers
    assert client.get("/api/health").status_code == 200       # GET needs no header


def test_host_allow_list(make_app):
    from fastapi.testclient import TestClient

    app = make_app(allowed_hosts=["127.0.0.1:8765", "localhost:8765"])
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        assert c.get("/api/health").status_code == 200
        r = c.get("/api/health", headers={"Host": "evil.example:8765"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
        r = c.get("/api/health", headers={"Host": "localhost:8765"})
        assert r.status_code == 200
    with TestClient(app) as c:                                # Host: testserver
        assert c.get("/api/health").status_code == 403


def test_websocket_foreign_origin_closed_1008(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", headers={"Origin": "http://evil.example"}):
            pass
    assert exc.value.code == 1008
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", headers={"Origin": "null"}):
            pass
    assert exc.value.code == 1008
    # same-origin and no-origin connections are accepted
    with client.websocket_connect("/ws", headers={"Origin": "http://testserver"}) as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        assert ws.receive_json()["type"] == "ws_ack"
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        assert ws.receive_json()["type"] == "ws_ack"


def test_websocket_host_check(make_app):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    with TestClient(make_app(allowed_hosts=["127.0.0.1:8765"])) as c:
        with pytest.raises(WebSocketDisconnect) as exc:
            with c.websocket_connect("/ws"):
                pass
        assert exc.value.code == 1008


@pytest.mark.parametrize("origin, host, ok", [
    ("http://127.0.0.1:8765", "127.0.0.1:8765", True),
    ("http://localhost", "localhost:80", True),
    ("https://localhost", "localhost:443", True),
    ("http://[::1]:8765", "[::1]:8765", True),
    ("http://127.0.0.1:8766", "127.0.0.1:8765", False),
    ("http://evil.example:8765", "127.0.0.1:8765", False),
    ("null", "127.0.0.1:8765", False),
])
def test_origin_matches_host(origin, host, ok):
    pytest.importorskip("fastapi")
    from mliprun.webapp.server import origin_matches_host

    assert origin_matches_host(origin, host) is ok
