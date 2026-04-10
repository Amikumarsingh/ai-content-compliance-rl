"""Validate all API endpoints."""
from server.app import app
from fastapi.testclient import TestClient
import json, sys

c = TestClient(app)
passed = 0
failed = 0

def check(label, r, expected_status=200, check_reward=False):
    global passed, failed
    ok = r.status_code == expected_status
    reward = r.json().get("reward") if check_reward else None
    done   = r.json().get("done")
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    line = f"[{status}] {label}: HTTP {r.status_code}"
    if check_reward:
        line += f" | reward={reward} done={done}"
    print(line)
    return r.json()

# GET endpoints
check("GET /health",   c.get("/health"))
check("GET /metadata", c.get("/metadata"))
check("GET /schema",   c.get("/schema"))
check("GET /state",    c.get("/state"))

# POST /reset with empty body (validator sends this)
reset = check("POST /reset (empty body)", c.post("/reset", json={}))
content = reset.get("observation", {}).get("content", "")
print(f"       content preview: {content[:60]}")

# POST /reset with seed for reproducibility
reset2 = check("POST /reset (seed=42)", c.post("/reset", json={"seed": 42}))

# POST /step - all 4 steps in sequence
check("POST /step step1 detect",  c.post("/step", json={"action": {"action_type": "detect_violations", "metadata": {"violations": ["spam", "suspicious_link"]}}}), check_reward=True)
check("POST /step step2 score",   c.post("/step", json={"action": {"action_type": "score_compliance",  "metadata": {"score": 0.35}}}), check_reward=True)
check("POST /step step3 decide",  c.post("/step", json={"action": {"action_type": "edit"}}), check_reward=True)
r4 = check("POST /step step4 edit", c.post("/step", json={"action": {"action_type": "submit_edit", "edited_content": "Check out this opportunity for more information."}}), check_reward=True)
print(f"       done={r4.get('done')} (should be true)")

# POST /reset again then test approve flow
c.post("/reset", json={})
c.post("/step", json={"action": {"action_type": "detect_violations", "metadata": {"violations": []}}})
c.post("/step", json={"action": {"action_type": "score_compliance",  "metadata": {"score": 0.9}}})
c.post("/step", json={"action": {"action_type": "approve"}})
r_confirm = check("POST /step step4 confirm_approve", c.post("/step", json={"action": {"action_type": "confirm_approve"}}), check_reward=True)
print(f"       done={r_confirm.get('done')} (should be true)")

# WebSocket /ws endpoint
try:
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {}})
        msg = ws.receive_json()
        ws_ok = msg.get("type") == "observation"
        print(f"[{'PASS' if ws_ok else 'FAIL'}] WebSocket /ws reset: type={msg.get('type')}")
        if ws_ok:
            passed += 1
        else:
            failed += 1
        ws.send_json({"type": "close"})
except Exception as e:
    print(f"[FAIL] WebSocket /ws: {e}")
    failed += 1

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
