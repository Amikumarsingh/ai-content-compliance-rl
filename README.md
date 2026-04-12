---
title: AI Content Compliance OpenEnv
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - reinforcement-learning
  - content-moderation
  - openenv
  - fastapi
license: mit
---

# Content Compliance RL Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com)

A production-grade [OpenEnv](https://openenv.ai) environment for training AI agents on real-world content moderation tasks using Reinforcement Learning.

---

## Overview

Content moderation at scale is one of the hardest unsolved problems in platform safety. Human moderators burn out, rule-based filters are trivially bypassed, and static classifiers fail to generalize to new violation patterns.

This environment trains RL agents to moderate content the way an expert human would — by reasoning through violations, scoring compliance, making a decision, and refining when needed. Agents learn from reward signals across 140 real content samples spanning 9 violation types and 3 difficulty tiers.

**What makes this different from a classifier:**
- The agent reasons across 4 sequential steps per episode, not a single forward pass
- Partial credit rewards nuanced decisions, not just binary correct/wrong
- Difficulty-aware grading means easy cases are held to a higher standard than hard ones
- The edit workflow forces the agent to fix content, not just label it

---

## Task Design

Each episode runs a fixed 4-step workflow:

```
Step 1 — Violation Detection     detect which policy categories are violated
Step 2 — Compliance Scoring      assign a 0.0–1.0 compliance score
Step 3 — Moderation Decision     approve / reject / edit
Step 4 — Confirm or Rewrite      confirm the decision or submit cleaned content
```

This mirrors how real trust & safety teams operate: triage → score → decide → act.

---

## Dataset

**140 content samples** across 3 difficulty tiers, covering all 9 violation types:

| Tier | Samples | Description | Target Accuracy |
|------|---------|-------------|-----------------|
| Easy | 40 | Obvious violations, unambiguous clean content | ≥ 95% |
| Medium | 50 | Borderline cases, subtle harassment, edit-required content | ≥ 80% |
| Hard | 50 | Multi-violation, coded language, sarcasm traps, ambiguous context | ≥ 65% |

### Violation Types

| Type | Description |
|------|-------------|
| `hate_speech` | Language targeting groups with hate or slurs |
| `violence` | Threats or glorification of violence |
| `harassment` | Personal attacks, insults, demeaning language |
| `adult_content` | Explicit or sexually suggestive content |
| `misinformation` | False claims, conspiracy theories, health misinformation |
| `spam` | Promotional content, get-rich-quick schemes |
| `suspicious_link` | Shortened or obfuscated URLs |
| `engagement_bait` | Manipulative share/tag/comment requests |
| `illegal_content` | References to illegal goods, services, or activities |

### Hard Cases the Agent Must Handle

- **Sarcasm traps** — *"I will kill them with my cooking 😂"* → clean, not violence
- **Coded hate** — *"certain communities just don't fit our values"* → hate_speech without slurs
- **Satire + misinformation** — content that claims to be satire but spreads false claims
- **Context-dependent** — journalism about illegal activity vs. solicitation of it
- **Casual harassment** — *"absolute clown behaviour"* → harassment without explicit slurs

---

## Reward Structure

Rewards are shaped per step and scaled by difficulty. Each step contributes a weighted fraction of the episode reward.

| Step | Easy Weight | Medium Weight | Hard Weight |
|------|-------------|---------------|-------------|
| Violation Detection | 0.30 | 0.25 | 0.25–0.30 |
| Compliance Scoring | 0.20 | 0.20 | 0.15–0.25 |
| Moderation Decision | 0.35 | 0.30 | 0.30–0.35 |
| Confirm / Edit | 0.15 | 0.25 | 0.20–0.25 |

**Partial credit rules:**
- Detection uses F1 score against ground truth violations
- Scoring uses proximity to ground truth compliance score (tolerance widens with difficulty)
- Decision gives partial credit on hard/medium for reasonable alternatives (e.g. `edit` when `reject` expected)
- Edit quality is graded by how many violations remain in the rewritten content

**Penalties:**
- Approving content with severe violations (`hate_speech`, `violence`, `illegal_content`) → near-zero reward
- False positives on clean content → score penalty proportional to count

---

## Action Space

| Action | Type | Description |
|--------|------|-------------|
| `detect_violations` | Step 1 | Pass list of detected violation types via metadata |
| `score_compliance` | Step 2 | Pass float score 0.0–1.0 via metadata |
| `approve` | Step 3 | Content passes — no violations |
| `reject` | Step 3 | Content violates policy — remove |
| `edit` | Step 3 | Content fixable — rewrite required |
| `confirm_approve` | Step 4 | Confirm approve decision |
| `confirm_reject` | Step 4 | Confirm reject decision |
| `submit_edit` | Step 4 | Submit rewritten content |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The text content being evaluated |
| `violations` | `list[str]` | Violations revealed after step 1 |
| `compliance_score` | `float` | Score 0.0–1.0 |
| `stage` | `str` | `classification` / `decision` / `refinement` |
| `step_count` | `int` | Steps taken in current episode |
| `task_description` | `str` | Instruction for the current step |
| `feedback` | `str` | Grader feedback from previous action |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward from the last action |

---

## Installation

```bash
git clone https://github.com/Amikumarsingh/ai-content-compliance-rl.git
cd ai-content-compliance-rl

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY and optionally MODEL_NAME, ENV_BASE_URL
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` / `OPENAI_API_KEY` | Yes | — | API key for LLM inference |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | Model for agent inference |
| `ENV_BASE_URL` | No | `http://localhost:7860` | RL environment server URL |
| `DEBUG` | No | `false` | Enable verbose logging |

---

## Usage

### Run Inference Locally

```bash
python inference.py
```

Expected output:
```
[START] task=content_compliance_easy_001 env=content_compliance_env model=gpt-4o-mini
[STEP] step=1 action=detect(0_violations) reward=0.48 done=false error=null
[STEP] step=2 action=score(0.95) reward=0.32 done=false error=null
[STEP] step=3 action=approve reward=0.56 done=false error=null
[STEP] step=4 action=confirm_approve reward=0.24 done=true error=null
[END] success=true steps=4 score=0.4 rewards=0.48,0.32,0.56,0.24
```

### Run Against a Remote HF Space

```bash
ENV_BASE_URL=https://your-space.hf.space python inference.py
```

The inference script automatically health-checks the remote server before running episodes.

### Python API

```python
from environment import ContentComplianceEnvironment
from models import ContentAction

env = ContentComplianceEnvironment(difficulty="mixed")
obs = env.reset()

# Step 1: detect violations
action = ContentAction(action_type="detect_violations", metadata={"violations": ["spam"]})
obs = env.step(action)

# Step 2: score compliance
action = ContentAction(action_type="score_compliance", metadata={"score": 0.3})
obs = env.step(action)

# Step 3: decide
action = ContentAction(action_type="reject")
obs = env.step(action)

# Step 4: confirm
action = ContentAction(action_type="confirm_reject")
obs = env.step(action)

print(obs.reward, obs.done)
```

### Docker

```bash
docker build -t content-compliance-rl .

docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  content-compliance-rl

curl http://localhost:7860/health
```

---

## API Endpoints

The server exposes a fully OpenEnv-compatible REST + WebSocket API.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health and uptime |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action and observation schema |
| `GET` | `/state` | Current environment state |
| `GET` | `/spec` | OpenEnv environment specification |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `WS` | `/ws` | WebSocket for persistent sessions |

### Endpoint Validation

All endpoints are validated via `validate_endpoints.py`. Current status:

```
[PASS] GET /health
[PASS] GET /metadata
[PASS] GET /schema
[PASS] GET /state
[PASS] POST /reset (empty body)
[PASS] POST /reset (seed=42)
[PASS] POST /step step1 detect
[PASS] POST /step step2 score
[PASS] POST /step step3 decide
[PASS] POST /step step4 edit
[PASS] POST /step step4 confirm_approve
[PASS] WebSocket /ws
Results: 12 passed, 0 failed
```

Run locally:

```bash
python validate_endpoints.py
```

---

## Project Structure

```
content-compliance-rl/
├── environment.py          # Core RL environment (OpenEnv compliant)
├── models.py               # Pydantic models: Action, Observation, State
├── client.py               # WebSocket client for remote environments
├── inference.py            # Baseline inference script
├── data/
│   ├── content_samples.json   # 140 labeled content samples
│   └── ground_truth.json      # Deterministic ground truth + reward weights
├── graders/
│   ├── openai_evaluator.py    # OpenAI-based evaluation with fallback
│   ├── violation_detector.py  # Rule-based violation detection
│   └── edit_evaluator.py      # Edit quality grading
├── server/
│   └── app.py              # OpenEnv-core FastAPI server
├── hf_server.py            # Hugging Face Spaces server entry point
├── openenv.yaml            # OpenEnv environment specification
├── Dockerfile              # Container deployment
└── requirements.txt        # Python dependencies
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Easy accuracy | ≥ 95% | Binary correct/wrong, no partial credit |
| Medium accuracy | ≥ 80% | Partial credit for reasonable alternatives |
| Hard accuracy | ≥ 65% | Coded language, multi-violation, sarcasm |
| Overall episode reward | ≥ 0.60 | Normalized average across all steps |
| OpenAI evaluation success | ≥ 95% | Falls back to rule-based if API fails |

---

## Hugging Face Spaces Deployment

1. Create a new Space with Docker runtime
2. Push all files including `Dockerfile`
3. Add `OPENAI_API_KEY` (and optionally `MODEL_NAME`) as Space secrets
4. Space builds and deploys automatically
5. Set `ENV_BASE_URL=https://your-space.hf.space` when running inference

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: description'`
4. Push and open a Pull Request

When adding new content samples, ensure a matching entry exists in `ground_truth.json` with correct `violations`, `compliance_score`, `correct_action`, and `difficulty` fields.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built for the OpenEnv Hackathon 2026 · Demonstrates real-world utility in AI safety and platform governance*
