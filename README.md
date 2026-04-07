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

# Content Moderation RL Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com)

A production-grade OpenEnv environment for training AI agents on real-world content moderation tasks.

## 🎯 Overview

This environment simulates content moderation workflows where AI agents must:

1. **Analyze** user-generated content for policy violations
2. **Detect** hate speech, spam, harassment, misinformation, and other violations
3. **Decide** whether to approve, reject, or edit content
4. **Learn** from feedback with shaped rewards

### Key Features

- **OpenAI-powered evaluation** with graceful fallback to rule-based detection
- **Three difficulty levels**: easy (obvious cases), medium (borderline), hard (multi-violation)
- **Deterministic grading** with partial credit for near-correct actions
- **Production-ready** with Docker support and Hugging Face Spaces deployment
- **Strict logging format** for reproducibility: `[START]`, `[STEP]`, `[END]`

## 📦 Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/content-moderation-rl.git
cd content-moderation-rl

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Environment Variables

```bash
# Required for OpenAI evaluation
OPENAI_API_KEY=sk-...

# Optional configuration
LLM_MODEL=gpt-4o-mini       # Model for content evaluation
API_BASE_URL=https://api.openai.com/v1  # Custom API endpoint
DEBUG=false                  # Enable debug logging
```

## 🚀 Usage

### Python API

```python
from env_openenv import ContentModerationEnv

# Create environment
env = ContentModerationEnv(
    max_steps=5,
    difficulty="mixed",  # easy, medium, hard, mixed
    evaluator_provider="openai"
)

# Reset episode
obs = env.reset()
print(f"Content: {obs.content}")
print(f"Violations: {obs.violations}")
print(f"Score: {obs.score}")

# Take actions
from app.models import Action

action = Action(action_type="reject")
result = env.step(action)

print(f"Reward: {result.reward.value}")
print(f"Done: {result.done}")
```

### Run Inference

```bash
# Run baseline inference with OpenAI
python inference.py

# Expected output format:
# [START] Content Compliance Inference
# [STEP] easy_001: Evaluating content...
# [STEP] easy_001: action=reject, correct=True, reward=0.700
# [END] Average reward: 0.680
```

### Docker Deployment

```bash
# Build Docker image
docker build -t content-moderation-rl .

# Run container
docker run -p 8000:8000 \
    -e OPENAI_API_KEY=sk-... \
    content-moderation-rl

# Test health endpoint
curl http://localhost:8000/health
```

### Hugging Face Spaces

1. Create new Space with Docker runtime
2. Upload all files including `Dockerfile`
3. Add `OPENAI_API_KEY` as Space secret
4. Space automatically deploys

## 📋 Environment Specification

### Action Space

| Action | Description | Terminal |
|--------|-------------|----------|
| `approve` | Content passes compliance | Yes |
| `reject` | Content violates policies | Yes |
| `edit` | Modify content to fix violations | No |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `content` | str | Text content being evaluated |
| `violations` | list[str] | Detected violation types |
| `score` | float | Compliance score (0.0-1.0) |
| `step_count` | int | Current step number |
| `stage` | str | Current stage (classification/decision/refinement) |
| `previous_action` | str\|None | Last action taken |
| `action_history` | list[str] | All actions in episode |
| `feedback` | str\|None | Environment feedback |

### Violation Types

- `hate_speech`: Language targeting groups with hate/slurs
- `violence`: Threats or glorification of violence
- `harassment`: Personal attacks, insults, demeaning language
- `adult_content`: Explicit or sexually suggestive content
- `misinformation`: False claims, conspiracy theories
- `spam`: Promotional content, excessive self-promotion
- `suspicious_link`: Shortened URLs or suspicious links
- `engagement_bait`: Manipulative engagement requests
- `illegal_content`: References to illegal activities

### Reward Structure

| Component | Value | Description |
|-----------|-------|-------------|
| Correct approve | +1.0 | Correctly approving compliant content |
| Correct reject | +1.0 | Correctly rejecting violating content |
| Partial edit | +0.5 | Editing when appropriate |
| Wrong action | 0.0-0.3 | Incorrect decision |
| Harmful approval | -0.2 | Approving dangerous content |
| Step penalty | -0.05 | Per extra step |

## 🎮 Task Difficulty

### Easy Tasks
- Obvious spam detection
- Clear hate speech identification
- Clean content approval
- Expected accuracy: 95%

### Medium Tasks
- Borderline promotional content
- Subtle harassment detection
- Edit-required content
- Expected accuracy: 80%

### Hard Tasks
- Multi-violation content
- Coded hate speech
- Misinformation with truth mixing
- Expected accuracy: 65%

## 📊 Logging Format

The inference script uses strict logging for reproducibility:

```
[START] Content Compliance Inference
[START] Model: gpt-4o-mini
[START] Tasks: ['easy', 'medium', 'hard']
[STEP] easy_001: Evaluating content...
[STEP] easy_001: action=approve, expected=approve, correct=True, reward=0.700
[END] Average reward: 0.680
```

## 🏗️ Architecture

```
content-moderation-rl/
├── env_openenv.py        # Main OpenEnv environment
├── models.py             # Pydantic models (Action, Observation, Reward)
├── tasks/
│   └── tasks.py          # Task definitions
├── graders/
│   ├── openai_evaluator.py  # OpenAI-based evaluation
│   └── graders.py        # Deterministic scoring
├── utils/
│   ├── content_generator.py # Dynamic content generation
│   └── data_loader.py    # Data loading utilities
├── inference.py          # Baseline inference script
├── server/
│   └── app.py            # FastAPI OpenEnv server
├── openenv.yaml          # Environment specification
├── Dockerfile            # Docker deployment
└── requirements.txt      # Python dependencies
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Validate OpenEnv spec
openenv validate

# Test environment
python -c "from env_openenv import ContentModerationEnv; env = ContentModerationEnv(); obs = env.reset(); print(obs)"
```

## 📈 Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Easy accuracy | ≥95% | Correct decisions on obvious cases |
| Medium accuracy | ≥80% | Correct decisions on borderline content |
| Hard accuracy | ≥65% | Correct decisions on complex content |
| Overall reward | ≥0.6 | Average normalized reward |
| OpenAI success rate | ≥95% | Successful API evaluations |

## 🔧 Configuration

### openenv.yaml

```yaml
name: content-moderation-rl
version: 1.0.0
tasks:
  easy:
    max_steps: 2
  medium:
    max_steps: 3
  hard:
    max_steps: 5
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE for details.

## 🙏 Acknowledgments

- OpenEnv team for the environment specification
- OpenAI for content evaluation capabilities
- Content moderation research community

## 📞 Support

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: team@example.com

---

**Built for the OpenEnv Hackathon 2026**

This environment demonstrates real-world utility in content moderation, a critical application for AI safety and platform governance.
