# Content Moderation RL Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces)

A production-grade OpenEnv environment for training AI agents on real-world content moderation tasks.

## Overview

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

## Quick Start

### Local Development

```bash
# Clone and navigate to the project
cd hf_setup

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 7860

# Test the API
curl http://localhost:7860/health
```

### Environment Variables

```bash
# Required for OpenAI evaluation
OPENAI_API_KEY=sk-...

# Optional configuration
LLM_MODEL=gpt-4o-mini       # Model for content evaluation
API_BASE_URL=https://api.openai.com/v1  # Custom API endpoint
DEBUG=false                  # Enable debug logging
PORT=7860                    # Server port
```

## Usage

### Python API

```python
from env import ContentComplianceEnv
from models import Action

# Create environment
env = ContentComplianceEnv(
    max_steps=5,
    difficulty="mixed",  # easy, medium, hard, mixed
)

# Reset episode
obs = env.reset()
print(f"Content: {obs.content}")
print(f"Violations: {obs.violations}")
print(f"Score: {obs.score}")

# Take actions
action = Action(action_type="reject")
observation, reward, done, truncated, info = env.step(action)

print(f"Reward: {reward.value}")
print(f"Done: {done}")
```

### REST API

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d '{"difficulty": "easy", "max_steps": 2}'

# Take a step
curl -X POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d '{"action_type": "reject", "env_id": "abc123"}'

# Get current state
curl http://localhost:7860/state

# Get environment specification
curl http://localhost:7860/spec
```

### Docker Deployment

```bash
# Build Docker image
docker build -t content-moderation-rl .

# Run container
docker run -p 7860:7860 \
    -e OPENAI_API_KEY=sk-... \
    content-moderation-rl

# Test health endpoint
curl http://localhost:7860/health
```

### Hugging Face Spaces Deployment

1. Create new Space at https://huggingface.co/spaces
2. Select **Docker** as the runtime
3. Upload all files from `hf_setup/` directory
4. Add `OPENAI_API_KEY` as Space secret (Settings → Secrets)
5. Space automatically deploys

Your space will be available at: `https://huggingface.co/spaces/your-username/content-moderation-rl`

## Environment Specification

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
| Correct action | +0.7 | Correct approve/reject decision |
| Good edit | +0.8 | Successful content edit |
| Partial detection | +0.3-0.6 | Partial violation detection |
| Wrong action | +0.0-0.2 | Incorrect decision |
| Harmful approval | -0.5 | Approving dangerous content |
| Step penalty | -0.05 | Per extra step |

## Task Difficulty

### Easy Tasks
- Obvious spam detection
- Clear hate speech identification
- Clean content approval
- **Expected accuracy**: 95%

### Medium Tasks
- Borderline promotional content
- Subtle harassment detection
- Edit-required content
- **Expected accuracy**: 80%

### Hard Tasks
- Multi-violation content
- Coded hate speech
- Misinformation with truth mixing
- **Expected accuracy**: 65%

## Logging Format

The environment uses strict logging for reproducibility:

```
[START] env_id=abc123, difficulty=easy, score=0.90
[STEP] env_id=abc123, step=1, action=reject, reward=0.700, done=True
[STEP] env_id=abc123, step=1, action=edit, score=0.80, reward=0.880
[STEP] env_id=abc123, step=2, final=approve, score=0.85, reward=0.700
```

## Architecture

```
hf_setup/
├── app.py                  # FastAPI server with OpenEnv endpoints
├── env.py                  # ContentComplianceEnv implementation
├── models.py               # Pydantic models (Observation, Action, Reward)
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Docker deployment
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Dependencies

- `graders/openai_evaluator.py` - OpenAI-based content evaluation
- `graders/graders.py` - Deterministic scoring logic

## Testing

```bash
# Test environment creation
python -c "from env import ContentComplianceEnv; env = ContentComplianceEnv(); obs = env.reset(); print(obs)"

# Test API endpoints
curl http://localhost:7860/health
curl http://localhost:7860/spec

# Validate OpenEnv spec
openenv validate
```

## Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Easy accuracy | ≥95% | Correct decisions on obvious cases |
| Medium accuracy | ≥80% | Correct decisions on borderline content |
| Hard accuracy | ≥65% | Correct decisions on complex content |
| Average reward | ≥0.6 | Average normalized reward |
| OpenAI success rate | ≥95% | Successful API evaluations |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Get current state |
| `/spec` | GET | Environment specification |
| `/metrics` | GET | Server metrics |

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see LICENSE for details.

## Acknowledgments

- OpenEnv team for the environment specification
- OpenAI for content evaluation capabilities
- Content moderation research community

---

**Built for the OpenEnv Hackathon 2026**

This environment demonstrates real-world utility in content moderation, a critical application for AI safety and platform governance.
