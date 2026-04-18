# What This Project Is — In Simple Words

---

## The Real-World Problem

Every day, millions of people post things on the internet — comments, messages, reviews, posts.

Some of that content is harmful:
- Hate speech targeting people
- Spam trying to scam you
- Misinformation spreading false claims
- Harassment attacking individuals
- Links to dangerous or illegal things

**Someone (or something) has to review all of it.**

Right now, platforms use a mix of:
- Human moderators — who burn out, make inconsistent decisions, and can't keep up with volume
- Simple keyword filters — which are easily bypassed ("k1ll" instead of "kill")
- Basic classifiers — which give a yes/no answer with no reasoning

**None of these work well at scale.**

---

## The Solution This Project Builds

This project trains an **AI agent** to moderate content the way a skilled human moderator would — by *reasoning through it step by step*, not just pattern-matching.

The agent doesn't just say "bad" or "good". It:

1. **Detects** what specific violations are present
2. **Scores** how compliant the content is (0.0 = terrible, 1.0 = clean)
3. **Decides** what to do — approve it, reject it, or fix it
4. **Acts** — either confirms the decision or rewrites the content to remove violations

This is called a **4-step workflow**, and it mirrors exactly how real trust & safety teams operate.

---

## What is Reinforcement Learning (RL)?

Think of it like training a dog.

- The dog does something → you give it a treat (reward) or say "no" (penalty)
- Over thousands of repetitions, the dog learns what gets treats

In this project:
- The **agent** is the AI making moderation decisions
- The **environment** is this codebase — it presents content and grades decisions
- The **reward** is a score (0.01 to 0.99) that tells the agent how good its decision was
- Over many episodes, the agent learns to make better and better decisions

---

## What is OpenEnv?

OpenEnv is a standard framework (like a rulebook) for building RL environments.

It defines:
- How the environment exposes itself via an API (`/reset`, `/step`, `/health`)
- What an "observation" looks like (what the agent sees)
- What an "action" looks like (what the agent can do)
- How rewards are structured

This project is **fully OpenEnv-compliant**, meaning any RL agent that speaks OpenEnv can train on it — not just the one included here.

---

## How One Episode Works

An **episode** is one complete moderation task. Here's what happens:

```
Environment picks a piece of content from 140 real samples
        ↓
STEP 1 — Agent detects violations
         "I see: spam, suspicious_link"
         Environment grades: how accurate was that? → reward
        ↓
STEP 2 — Agent scores compliance
         "I give this a 0.3 out of 1.0"
         Environment grades: how close to ground truth? → reward
        ↓
STEP 3 — Agent decides
         "I'll edit this — it's fixable"
         Environment grades: was that the right call? → reward
        ↓
STEP 4 — Agent acts
         "Here's the rewritten version without the spam"
         Environment grades: did the edit actually fix it? → reward
        ↓
Episode ends. Total reward calculated.
```

The agent learns: decisions that get high rewards → do more of those.

---

## The 140 Content Samples

The dataset has 3 difficulty tiers:

| Tier | Count | What it looks like |
|------|-------|--------------------|
| Easy | 40 | "I hate those people. They should all die!" → obvious hate speech |
| Medium | 50 | "Some people are just too ignorant..." → subtle harassment |
| Hard | 50 | "Wake up sheeple! Visit bit.ly/truth" → coded language + multiple violations |

**Hard cases are designed to trick the agent:**
- *"I will kill them with my cooking 😂"* → NOT violence (sarcasm)
- *"Certain communities just don't fit our values"* → IS hate speech (no slurs, but coded)
- Journalism about illegal activity → NOT illegal content
- Satire that spreads false claims → IS misinformation

---

## How Rewards Work

The reward system is designed to teach the agent *good judgment*, not just rule-following.

| What the agent does | Reward |
|---------------------|--------|
| Correctly rejects hate speech | High (~0.90) |
| Correctly approves clean content | High (~0.95) |
| Edits fixable spam and cleans it | Medium (~0.75) |
| Approves content with hate speech | Very low (~0.02) — worst mistake |
| Rejects clean content (false alarm) | Low (~0.15) |
| Takes too many steps to decide | Penalty per extra step |
| Repeats the same action in a loop | Penalty |

**Difficulty changes the grading:**
- Easy: must get it exactly right — no excuses
- Medium: partial credit for reasonable alternatives
- Hard: even missing 1-2 violations still gets partial credit (it's genuinely hard)

---

## The Codebase — What Each File Does

```
inference.py          The agent script — uses GPT-4o-mini to make decisions
                      Logs [START]/[STEP]/[END] for the hackathon validator

environment.py        The core RL environment
                      Loads 140 content samples, grades every step,
                      returns rewards based on ground truth

models.py             Data types — what an Action and Observation look like
                      (ContentAction, ContentObservation, ContentState)

server/app.py         The REST API server
                      Wraps the environment so any agent can call it over HTTP

hf_server.py          Alternative server (simpler, no openenv-core dependency)
                      Used as fallback

data/
  content_samples.json    140 real content samples with difficulty labels
  ground_truth.json       Correct answers: violations, score, right action

graders/
  violation_detector.py   Rule-based scanner — finds violations by pattern
  edit_evaluator.py       Grades how good an edit was
  openai_evaluator.py     Uses GPT to evaluate content (more accurate)

app/
  env.py              Alternative environment implementation
  reward.py           Advanced reward calculator with bonuses/penalties
  models.py           Pydantic models for the app layer

openenv.yaml          The environment specification — like a contract
                      Defines action space, observation space, reward range

Dockerfile            Packages everything for deployment
start.sh              Starts the server on port 7860
pyproject.toml        Python package definition (required by openenv validate)
uv.lock               Locked dependencies (required by openenv validate)
```

---

## Why This Matters Beyond the Hackathon

Content moderation is one of the hardest unsolved problems in platform safety.

This project demonstrates that:

1. **RL can learn nuanced judgment** — not just binary classification
2. **Multi-step reasoning beats single-pass models** — the agent thinks before deciding
3. **Difficulty-aware grading produces better agents** — easy cases are held to higher standards
4. **The edit workflow is critical** — fixing content is often better than removing it

A production version of this could:
- Reduce human moderator workload by handling obvious cases automatically
- Flag borderline cases for human review with reasoning attached
- Adapt to new violation patterns through continued training
- Explain its decisions (step 1 detection + step 2 score = transparent reasoning)

---

## The Hackathon Submission Requirements

This project satisfies all of them:

| Requirement | How it's met |
|-------------|--------------|
| Real-world task | Content moderation — not a game or toy |
| OpenEnv spec compliance | `environment.py` extends `Environment`, typed models, `openenv.yaml` |
| 3+ tasks with graders | Easy / Medium / Hard with deterministic grading |
| Meaningful reward function | 4-step partial credit, difficulty-aware, F1 scoring |
| Baseline inference script | `inference.py` — runs 8 tasks, logs exact format |
| HF Spaces deployment | Docker + `start.sh` on port 7860 |
| Dockerfile | Present at repo root |
| README | Full documentation |
| Scores strictly (0, 1) | `_clamp()` ensures no exact 0.0 or 1.0 |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | All three mandatory vars used |
| OpenAI client for LLM calls | `from openai import OpenAI` used directly |
| `openenv validate` passes | `pyproject.toml`, `uv.lock`, `server/app.py` all present |

---

## How to Run It

**Locally:**
```bash
# Install dependencies
pip install -r requirements.txt

# Add your API key to .env
# HF_TOKEN=sk-...

# Run the agent
python inference.py

# Start the server
python hf_server.py
```

**With Docker:**
```bash
docker build -t content-compliance-rl .
docker run -p 7860:7860 -e HF_TOKEN=sk-... content-compliance-rl
```

**Test the server:**
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset
```

---

## Sample Output

```
[START] task=content_compliance_easy_001 env=content_compliance_env model=gpt-4o-mini
[STEP] step=1 action=approve reward=0.95 done=true error=null
[END] success=true steps=1 score=0.95 rewards=0.95

[START] task=content_compliance_medium_001 env=content_compliance_env model=gpt-4o-mini
[STEP] step=1 action=edit reward=0.45 done=false error=null
[STEP] step=2 action=approve reward=0.9 done=true error=null
[END] success=true steps=2 score=0.675 rewards=0.45,0.9

Baseline average score: 0.814 / 1.0
```

Easy tasks: 1 step, high reward.
Medium tasks: 2 steps (edit then approve), varied rewards.
Hard tasks: 1-3 steps, lower rewards reflecting genuine difficulty.

---

*Built for the OpenEnv Hackathon 2026 — demonstrating real-world utility in AI safety and platform governance.*
