"""Test full 4-step WebSocket episode."""
from server.app import app
from fastapi.testclient import TestClient

c = TestClient(app)

with c.websocket_connect("/ws") as ws:
    # Reset
    ws.send_json({"type": "reset", "data": {}})
    msg = ws.receive_json()
    content = msg["data"].get("content", "")[:60]
    print(f"[PASS] WS reset: content={content}")

    # Step 1
    ws.send_json({"type": "step", "data": {"action_type": "detect_violations", "metadata": {"violations": ["spam"]}}})
    msg = ws.receive_json()
    r1 = msg["data"].get("reward")
    print(f"[PASS] WS step1: reward={r1} done={msg['data'].get('done')}")

    # Step 2
    ws.send_json({"type": "step", "data": {"action_type": "score_compliance", "metadata": {"score": 0.4}}})
    msg = ws.receive_json()
    r2 = msg["data"].get("reward")
    print(f"[PASS] WS step2: reward={r2} done={msg['data'].get('done')}")

    # Step 3
    ws.send_json({"type": "step", "data": {"action_type": "reject"}})
    msg = ws.receive_json()
    r3 = msg["data"].get("reward")
    print(f"[PASS] WS step3: reward={r3} done={msg['data'].get('done')}")

    # Step 4
    ws.send_json({"type": "step", "data": {"action_type": "confirm_reject"}})
    msg = ws.receive_json()
    r4 = msg["data"].get("reward")
    done = msg["data"].get("done")
    status = "PASS" if done else "FAIL"
    print(f"[{status}] WS step4: reward={r4} done={done} (should be True)")

    ws.send_json({"type": "close"})

print(f"\nAll rewards: {r1}, {r2}, {r3}, {r4}")
print(f"Avg: {round((r1+r2+r3+r4)/4, 3)}")
