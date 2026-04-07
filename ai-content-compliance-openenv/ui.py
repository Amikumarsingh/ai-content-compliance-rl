"""
Gradio UI for Content Compliance RL Environment.
Provides Quick Start guide and interactive Playground.
"""

import gradio as gr
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
import os

# ── API helpers ────────────────────────────────────────────────────────────────

def api_reset(difficulty):
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"difficulty": difficulty, "max_steps": 5}, timeout=15)
        return r.json()
    except Exception as e:
        return {"status": "error", "info": {"error": str(e)}}

def api_step(action_type, edited_content=None):
    try:
        payload = {"action_type": action_type}
        if edited_content and edited_content.strip():
            payload["edited_content"] = edited_content
        r = requests.post(f"{BASE_URL}/step", json=payload, timeout=15)
        return r.json()
    except Exception as e:
        return {"status": "error", "info": {"error": str(e)}}

# ── Session state ──────────────────────────────────────────────────────────────

session = {"obs": None, "done": False, "total_reward": 0.0, "steps": 0, "history": []}

def _pack(content, violations, score, reward, steps, status, interactive, history):
    """Pack outputs into the 10-element tuple expected by Gradio."""
    return (
        content, violations, score, reward, steps, status,
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
        gr.update(interactive=interactive),
        history,
    )

def reset_session(difficulty):
    result = api_reset(difficulty)
    if result.get("status") == "error":
        err = result.get("info", {}).get("error", "Unknown error")
        return _pack(f"❌ Cannot connect to backend: {err}", "—", "—", "0.00", "0",
                     "❌ Backend unreachable — is the server running?", False, "")

    obs = result.get("observation", {})
    session.update(obs=obs, done=False, total_reward=0.0, steps=0, history=[])

    violations = obs.get("violations", [])
    score      = obs.get("score", 0.0)
    content    = obs.get("content", "(no content)")
    viol_str   = ", ".join(violations) if violations else "✅ None detected"
    hint       = "🟢 Looks clean — consider Approve" if not violations else f"🔴 {len(violations)} violation(s) — consider Reject or Edit"

    return _pack(content, viol_str, f"{score:.2f}", "0.00", "0",
                 f"🟢 New episode loaded! {hint}", True, "")

def take_action(action_type, edited_text):
    prev_content   = session["obs"].get("content", "") if session["obs"] else ""
    prev_history   = "\n".join(session["history"])
    prev_reward    = f"{session['total_reward']:.3f}"
    prev_steps     = str(session["steps"])

    if session["done"] or session["obs"] is None:
        return _pack(prev_content, "", "", prev_reward, prev_steps,
                     "⚠️ Episode done — press 🔄 New Episode to play again.", False, prev_history)

    result = api_step(action_type, edited_text if action_type == "edit" else None)

    if result.get("status") == "error":
        err = result.get("info", {}).get("error", "Unknown")
        return _pack(prev_content, "", "", prev_reward, prev_steps,
                     f"❌ Error: {err}", True, prev_history)

    obs        = result.get("observation", {})
    reward_obj = result.get("reward", {})
    reward     = reward_obj.get("value", 0.0)
    done       = result.get("done", False)

    session["obs"]           = obs
    session["done"]          = done
    session["total_reward"] += reward
    session["steps"]        += 1

    violations  = obs.get("violations", [])
    score       = obs.get("score", 0.0)
    content     = obs.get("content", prev_content)
    viol_str    = ", ".join(violations) if violations else "✅ None"
    explanation = reward_obj.get("explanation", "")
    emoji       = "🟢" if reward >= 0.7 else "🟡" if reward >= 0.4 else "🔴"

    session["history"].append(
        f"Step {session['steps']}: {action_type.upper()} {emoji} reward={reward:.3f} | {explanation}"
    )
    history_text = "\n".join(session["history"])

    if done:
        avg   = session["total_reward"] / session["steps"]
        grade = "🏆 Excellent!" if avg >= 0.8 else "👍 Good" if avg >= 0.6 else "💪 Keep practicing"
        return _pack(content, viol_str, f"{score:.2f}",
                     f"{session['total_reward']:.3f}", str(session["steps"]),
                     f"✅ Episode done! Avg reward: {avg:.3f} — {grade}", False, history_text)

    hint = "🟢 Looks clean now — Approve?" if not violations else f"🔴 Still {len(violations)} violation(s) — Reject or Edit"
    return _pack(content, viol_str, f"{score:.2f}",
                 f"{session['total_reward']:.3f}", str(session["steps"]),
                 f"🔄 Step {session['steps']} done. {hint}", True, history_text)

def approve():       return take_action("approve", None)
def reject():        return take_action("reject",  None)
def edit(txt):       return take_action("edit",    txt)

def moderate_custom(custom_text, difficulty):
    """Load user-pasted content into the environment."""
    if not custom_text or not custom_text.strip():
        return _pack("", "", "", "0.00", "0",
                     "⚠️ Please paste some content first.", False, "")
    try:
        # Reset env with given difficulty
        result = api_reset(difficulty)
        if result.get("status") == "error":
            err = result.get("info", {}).get("error", "Unknown")
            return _pack("", "", "", "0.00", "0", f"❌ Backend error: {err}", False, "")

        # Override the content in session with user's text
        obs = result.get("observation", {})
        obs["content"] = custom_text.strip()

        # Detect violations locally for display
        text_lower = custom_text.lower()
        violations = []
        if any(w in text_lower for w in ["hate", "slur", "racist", "bigot"]):
            violations.append("hate_speech")
        if any(w in text_lower for w in ["kill", "die", "threat", "hurt", "violence"]):
            violations.append("violence")
        if any(w in text_lower for w in ["idiot", "stupid", "loser", "ugly", "dumb"]):
            violations.append("harassment")
        if any(w in text_lower for w in ["buy now", "click here", "bit.ly", "tinyurl", "limited offer", "free money"]):
            violations.append("spam")
        if any(w in text_lower for w in ["5g", "conspiracy", "they don't want you", "fake news", "hoax"]):
            violations.append("misinformation")
        if any(w in text_lower for w in ["sex", "nude", "explicit", "adult"]):
            violations.append("adult_content")
        if any(w in text_lower for w in ["illegal", "drugs", "weapon", "hack"]):
            violations.append("illegal_content")

        score = 0.2 if violations else 0.9
        obs["violations"] = violations
        obs["score"] = score

        session.update(obs=obs, done=False, total_reward=0.0, steps=0, history=[])

        viol_str = ", ".join(violations) if violations else "✅ None detected"
        hint = f"🔴 {len(violations)} violation(s) found — Reject or Edit" if violations else "🟢 Looks clean — Approve"

        return _pack(custom_text.strip(), viol_str, f"{score:.2f}", "0.00", "0",
                     f"🔍 Custom content loaded! {hint}", True, "")
    except Exception as e:
        return _pack("", "", "", "0.00", "0", f"❌ Error: {e}", False, "")

# ── Quick Start content ────────────────────────────────────────────────────────

QUICK_START_MD = """
## 🚀 Quick Start

### What is this?
A **Reinforcement Learning environment** for training AI agents on content moderation.
Your agent reads user-generated content and decides whether to **approve**, **reject**, or **edit** it.

### 🎮 Try it yourself → Go to the **Playground** tab!

---

### Actions
| Action | When to use | Reward |
|--------|-------------|--------|
| ✅ Approve | Content is clean and safe | +1.0 |
| ❌ Reject  | Content violates policies | +1.0 |
| ✏️ Edit    | Content needs minor fixes | +0.7 |

### Violation Types
| Type | Example |
|------|---------|
| `hate_speech` | Slurs or language targeting groups |
| `violence` | Threats or glorification of harm |
| `harassment` | Personal attacks, insults |
| `spam` | Promotional content, clickbait |
| `misinformation` | False claims, conspiracy theories |
| `adult_content` | Explicit or suggestive content |
| `illegal_content` | References to illegal activities |

### Difficulty Levels
| Level | Description | Expected Accuracy |
|-------|-------------|------------------|
| 🟢 Easy   | Obvious spam or clean content | ~95% |
| 🟡 Medium | Borderline or promotional content | ~80% |
| 🔴 Hard   | Multi-violation, coded language | ~65% |

### Reward Structure
```
Correct approve/reject  → +1.0
Good edit (fixes all)   → +0.7
Partial edit            → +0.3
False positive          → -0.5
False negative          → -1.0  (worst!)
Per step cost           → -0.05
Early finish bonus      → +0.2
```

### REST API
```bash
# Reset environment
curl -X POST https://amikumarsingh-ai-content-compliance-openenv.hf.space/reset

# Take action
curl -X POST https://amikumarsingh-ai-content-compliance-openenv.hf.space/step \\
  -H "Content-Type: application/json" \\
  -d '{"action_type": "reject"}'
```

### Python
```python
import requests
BASE = "https://amikumarsingh-ai-content-compliance-openenv.hf.space"
obs  = requests.post(f"{BASE}/reset").json()
result = requests.post(f"{BASE}/step", json={"action_type": "reject"}).json()
print(result["reward"]["value"])
```
"""

# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Content Compliance RL") as demo:

    gr.Markdown("# 🛡️ Content Compliance RL Environment")
    gr.Markdown("Train and test AI agents on real-world content moderation tasks.")

    with gr.Tabs():

        # Tab 1: Quick Start
        with gr.Tab("📖 Quick Start"):
            gr.Markdown(QUICK_START_MD)

        # Tab 2: Playground
        with gr.Tab("🎮 Playground"):
            gr.Markdown("""
### How to play
1. Choose a difficulty and click **🔄 New Episode** to load content
2. Read the content and check detected violations
3. Click **✅ Approve** if safe, **❌ Reject** if it violates policy,
   or paste a fixed version below and click **✏️ Edit**
4. Earn rewards and try to score as high as possible!
            """)

            with gr.Row():
                difficulty = gr.Dropdown(["easy", "medium", "hard", "mixed"],
                                         value="mixed", label="Difficulty", scale=1)
                reset_btn  = gr.Button("🔄 New Episode", variant="primary", scale=2)

            with gr.Row():
                custom_content = gr.Textbox(
                    label="📝 Or paste your own content to moderate (optional)",
                    placeholder="Paste any text here and click 'Moderate My Content' to test it...",
                    lines=3, scale=3
                )
                custom_btn = gr.Button("🔍 Moderate My Content", variant="secondary", scale=1)

            status_box = gr.Textbox(label="Status", interactive=False,
                                    value="Press '🔄 New Episode' or paste your own content")

            with gr.Row():
                with gr.Column(scale=2):
                    content_box = gr.Textbox(label="📄 Content to Moderate", lines=5,
                                             interactive=False,
                                             placeholder="Click '🔄 New Episode' to load content...")
                    edit_box    = gr.Textbox(label="✏️ Edited Content (for Edit action)", lines=3,
                                             placeholder="Paste your cleaned version here, then click ✏️ Edit...")

                with gr.Column(scale=1):
                    violations_box = gr.Textbox(label="⚠️ Violations Detected", interactive=False,
                                                placeholder="Appears after New Episode")
                    score_box      = gr.Textbox(label="📊 Compliance Score (0.0 – 1.0)",
                                                interactive=False, placeholder="0.00")
                    reward_box     = gr.Textbox(label="🏆 Total Reward", interactive=False, value="0.00")
                    steps_box      = gr.Textbox(label="👣 Steps Taken",  interactive=False, value="0")

            with gr.Row():
                approve_btn = gr.Button("✅ Approve — Content is safe",  variant="secondary", interactive=False)
                reject_btn  = gr.Button("❌ Reject — Policy violation",   variant="stop",      interactive=False)
                edit_btn    = gr.Button("✏️ Edit — Fix then resubmit",   variant="secondary", interactive=False)

            gr.Markdown("_Tip: Score ≥ 0.5 with no violations → Approve. Any violations → Reject or Edit._")
            history_box = gr.Textbox(label="📜 Step History", lines=6, interactive=False)

            outs = [content_box, violations_box, score_box, reward_box, steps_box,
                    status_box, approve_btn, reject_btn, edit_btn, history_box]

            reset_btn.click(reset_session, inputs=[difficulty], outputs=outs)
            custom_btn.click(moderate_custom, inputs=[custom_content, difficulty], outputs=outs)
            approve_btn.click(approve, inputs=[],         outputs=outs)
            reject_btn.click(reject,   inputs=[],         outputs=outs)
            edit_btn.click(edit,       inputs=[edit_box], outputs=outs)

        # Tab 3: API Docs
        with gr.Tab("📡 API Docs"):
            gr.Markdown("""
## Interactive API Documentation

Full Swagger UI available at:

👉 [Open Swagger UI](/docs)

👉 [Open ReDoc](/redoc)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`       | API info |
| GET  | `/health` | Health check |
| POST | `/reset`  | Start new episode |
| POST | `/step`   | Take action |
| GET  | `/spec`   | OpenEnv spec |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
