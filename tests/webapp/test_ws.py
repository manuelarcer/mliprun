"""WebSocket /ws: replay-then-live with no gap or duplicate, '*' subscription, errors."""
import time

H = {"X-Mliprun": "1"}


def _recv_until(ws, predicate, max_messages=10000):
    """Receive messages (skipping pings) until ``predicate(msg)``; return them all."""
    got = []
    for _ in range(max_messages):
        msg = ws.receive_json()
        if msg.get("type") == "ws_ping":
            continue
        got.append(msg)
        if predicate(msg):
            return got
    raise AssertionError("predicate never satisfied")


def _is_final(job_id):
    return lambda m: (m.get("job_id") == job_id and m.get("type") == "job_status"
                      and m.get("status") in ("done", "failed", "cancelled"))


def test_replay_then_live_has_no_gap_or_duplicate(client, cu_upload, submit):
    job_id = submit(client, [cu_upload["id"]], label="steps=60;sleep=0.01") \
        .json()["jobs"][0]["job_id"]
    # wait until the job is mid-run, so part comes from replay and part live
    deadline = time.monotonic() + 20
    while client.get(f"/api/jobs/{job_id}/events").json()["next_seq"] < 10:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": 0})
        msgs = _recv_until(ws, _is_final(job_id))
    acks = [m for m in msgs if m["type"] == "ws_ack"]
    assert len(acks) == 1 and acks[0]["job_id"] == job_id and acks[0]["replayed"] >= 10
    events = [m for m in msgs if "seq" in m]
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs)))               # contiguous, no dup, no gap
    assert msgs.index(acks[0]) == acks[0]["replayed"]   # ack follows the replayed block
    assert events == client.get(f"/api/jobs/{job_id}/events").json()["events"]


def test_since_mid_stream_and_null(client, cu_upload, submit, wait_for_status):
    job_id = submit(client, [cu_upload["id"]], label="steps=3").json()["jobs"][0]["job_id"]
    wait_for_status(client, job_id, "done")
    n = client.get(f"/api/jobs/{job_id}/events").json()["next_seq"]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": n - 2})
        msgs = _recv_until(ws, lambda m: m["type"] == "ws_ack")
        assert [m["seq"] for m in msgs[:-1]] == [n - 2, n - 1]
        assert msgs[-1]["replayed"] == 2
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": None})
        ack = ws.receive_json()
        assert ack == {"type": "ws_ack", "action": "subscribe", "job_id": job_id,
                       "replayed": 0}


def test_star_subscription_is_light_and_deduplicated(client, cu_upload, submit):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        assert ws.receive_json()["type"] == "ws_ack"
        job_id = submit(client, [cu_upload["id"]], label="steps=5;sleep=0.01") \
            .json()["jobs"][0]["job_id"]
        star = _recv_until(ws, _is_final(job_id))
        assert {m["type"] for m in star} <= {"job_status", "result", "error"}
        assert [m["status"] for m in star if m["type"] == "job_status"] == \
            ["pending", "running", "done"]
        assert sum(m["type"] == "result" for m in star) == 1
        # now also subscribe to the job from 0: the light events already delivered
        # through "*" are not sent again
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": 0})
        msgs = _recv_until(ws, lambda m: m["type"] == "ws_ack")
        replayed = msgs[:-1]
        all_events = client.get(f"/api/jobs/{job_id}/events").json()["events"]
        star_seqs = {m["seq"] for m in star}
        assert [m["seq"] for m in replayed] == [e["seq"] for e in all_events
                                                if e["seq"] not in star_seqs]
        assert msgs[-1]["replayed"] == len(replayed)


def test_both_subscriptions_deliver_live_events_once(client, cu_upload, submit):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "subscribe", "job_id": "*"})
        ws.receive_json()
        job_id = submit(client, [cu_upload["id"]], label="steps=40;sleep=0.02") \
            .json()["jobs"][0]["job_id"]
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": 0})
        msgs = _recv_until(ws, _is_final(job_id))
    seqs = [m["seq"] for m in msgs if m.get("job_id") == job_id and "seq" in m]
    assert sorted(seqs) == list(range(len(seqs)))
    assert len(seqs) == len(set(seqs))


def test_unsubscribe_and_errors(client, cu_upload, submit, wait_for_status):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_json({"action": "dance", "job_id": "*"})
        assert ws.receive_json() == {"type": "ws_error", "code": "bad_request",
                                     "message": "action must be 'subscribe' or 'unsubscribe'"}
        ws.send_json({"action": "subscribe", "job_id": "20260101-000000-md-abcdef"})
        assert ws.receive_json()["code"] == "not_found"
        ws.send_json({"action": "subscribe", "job_id": "*"})
        ws.receive_json()
        ws.send_json({"action": "unsubscribe", "job_id": "*"})
        assert ws.receive_json()["action"] == "unsubscribe"
        job_id = submit(client, [cu_upload["id"]]).json()["jobs"][0]["job_id"]
        wait_for_status(client, job_id, "done")
        ws.send_json({"action": "subscribe", "job_id": job_id, "since": -3})
        msg = ws.receive_json()             # nothing from the job arrived in between
        assert msg["type"] == "ws_error" and msg["code"] == "bad_request"
