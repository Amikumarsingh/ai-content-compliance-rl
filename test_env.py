from server.app import app
from fastapi.testclient import TestClient
import json

c = TestClient(app)

r = c.post("/reset", json={})
r2 = c.post("/step", json={"action": {"action_type": "detect_violations", "metadata": {"violations": ["harassment"]}}})
r3 = c.post("/step", json={"action": {"action_type": "score_compliance", "metadata": {"score": 0.4}}})
r4 = c.post("/step", json={"action": {"action_type": "reject"}})
r5 = c.post("/step", json={"action": {"action_type": "confirm_reject"}})

with open("test_out.txt", "w", encoding="utf-8") as f:
    f.write("RESET: " + str(r.status_code) + "\n")
    d = r.json()
    f.write("content: " + d["observation"]["content"][:80] + "\n\n")
    for i, resp in enumerate([r2, r3, r4, r5], 1):
        d = resp.json()
        f.write(f"STEP {i}: status={resp.status_code} reward={d.get('reward')} done={d.get('done')}\n")
        f.write(f"  feedback: {d['observation']['feedback'][:80]}\n")
