---
title: AI Content Compliance RL
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - reinforcement-learning
  - content-moderation
  - openenv
license: mit
---

# Content Compliance RL Environment

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](https://github.com/anthropics/openenv)

A production-ready reinforcement learning environment for training content moderation agents.

Built on the OpenEnv specification with LLM-powered evaluation, dense reward signals, and progressive difficulty scaling.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo inference
python main.py demo

# Start the API server
python main.py serve

# Run tests
python main.py test
```

---

## Project Structure

```
ai-content-compliance-openenv/
├── app/                     # Core application module
│   ├── models.py           # Pydantic models (Observation, Action, Reward)
│   ├── env.py              # ContentComplianceEnv implementation
│   ├── reward.py           # RewardCalculator for dense rewards
│   └── state_manager.py    # Episode state tracking
├── tasks/                   # Task definitions
│   ├── task_1_easy_decision.py
│   ├── task_2_medium_edit.py
│   ├── task_3_hard_complex.py
│   └── __init__.py
├── graders/                 # Grading system
│   ├── grader.py           # Main grading logic
│   ├── violation_detector.py
│   ├── test_grader.py      # Unit tests
│   └── __init__.py
├── rewards/                 # Reward functions
│   ├── reward_function.py  # Dense, explainable rewards
│   ├── test_reward.py      # Unit tests
│   └── __init__.py
├── agents/                  # RL agents
│   └── q_agent.py          # Q-Learning with replay buffer
├── environments/            # Gymnasium-compatible wrappers
│   └── content_compliance_env.py
├── llm/                     # LLM integration
│   ├── evaluator.py        # Multi-agent evaluation
│   └── prompt_templates.py
├── openenv/                 # OpenEnv reference implementation
│   ├── models.py
│   └── environment.py
├── server/                  # FastAPI server
│   └── main.py
├── utils/                   # Utilities
│   ├── config.py
│   ├── data_loader.py
│   └── helpers.py
├── inference.py             # Inference script
├── main.py                  # CLI entry point
├── hf_server.py            # Hugging Face Spaces server
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Production Docker
├── Dockerfile.space        # HF Spaces Docker
├── docker-compose.yml      # Compose config
├── requirements.txt        # Dependencies
└── README.md
```

---

## OpenEnv Specification

### Action Space

| Action | Value | Description | Terminal |
|--------|-------|-------------|----------|
| `approve` | 0 | Content passes compliance review | Yes |
| `reject` | 1 | Content violates policies | Yes |
| `edit` | 2 | Content needs modification | No |

### Observation Space

```python
{
    "content": str,           # Text content being evaluated
    "violations": list[str],  # Detected violation types
    "score": float,           # Compliance score [0.0, 1.0]
    "step_count": int,        # Current step in episode
}
```

### Violation Types

- `hate_speech` - Language targeting groups with hate/slurs
- `violence` - Threats or glorification of violence
- `harassment` - Personal attacks, insults, demeaning language
- `adult_content` - Explicit or adult-themed content
- `misinformation` - False claims, conspiracy theories
- `spam` - Promotional, clickbait, excessive punctuation
- `illegal_content` - References to illegal activities

### Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Violation Detection | 30% | Correct identification of violations |
| Action Correctness | 30% | Choosing the right action |
| Edit Quality | 20% | Effectiveness of content edits |
| Efficiency | 20% | Completing in optimal steps |

---

## Usage

### Python API

```python
from openenv import ContentComplianceEnv, Action, Observation

# Create environment
env = ContentComplianceEnv(max_steps=5, difficulty="mixed")

# Reset and get initial observation
obs = await env.reset()
print(f"Content: {obs.content}")
print(f"Score: {obs.score}")

# Take actions
action = Action(action_type="approve")
result = await env.step(action)

print(f"Reward: {result.reward.value}")
print(f"Done: {result.done}")
```

### REST API

```bash
# Start server
python hf_server.py

# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"max_steps": 5, "difficulty": "mixed"}'

# Take action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "approve"}'
```

---

## Docker Deployment

### Build Image

```bash
docker build -t compliance-env .
```

### Run Container

```bash
# API server
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  compliance-env

# Health check
curl http://localhost:7860/health
```

### Docker Compose

```bash
# Start application
docker compose up app

# Run training
docker compose --profile train up train
```

---

## Hugging Face Deployment

### Quick Deploy

1. Create a new Space at https://huggingface.co/spaces/create
2. Select Docker SDK as the runtime
3. Set `Dockerfile.space` as the Dockerfile
4. Push your repository

```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli repo create your-username/content-compliance \
    --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/your-username/content-compliance
git push hf main
```

---

## Testing

```bash
# Run all tests
python main.py test

# Run specific test modules
pytest graders/test_grader.py -v
pytest rewards/test_reward.py -v
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key for LLM evaluation |
| `API_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `MODEL_NAME` | `gpt-4o` | Model for evaluation |
| `EVALUATOR_PROVIDER` | `mock` | Provider type (mock/openai) |
| `PORT` | `7860` | Server port |

---

## License

MIT License - see LICENSE for details.
