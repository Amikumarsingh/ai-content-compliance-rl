"""
Content Compliance RL Environment — OpenEnv compliant implementation.

Implements the Environment interface from openenv-core with:
- reset() / step() / state property
- 50 real content samples loaded from data/content_samples.json
- Deterministic grading against data/ground_truth.json
- 3 difficulty levels: easy, medium, hard
- 4-step workflow per episode with partial credit at every step
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import ContentAction, ContentObservation, ContentState

# ── Load data files ───────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_samples() -> Dict[str, List[dict]]:
    """Load content samples grouped by difficulty."""
    path = os.path.join(_DATA_DIR, "content_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    grouped: Dict[str, List[dict]] = {"easy": [], "medium": [], "hard": []}
    for s in samples:
        grouped[s["difficulty"]].append(s)
    return grouped


def _load_ground_truth() -> Dict[str, dict]:
    """Load ground truth keyed by sample id."""
    path = os.path.join(_DATA_DIR, "ground_truth.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SAMPLES = _load_samples()
GROUND_TRUTH = _load_ground_truth()

# ── Violation taxonomy ────────────────────────────────────────────────────────

SEVERE   = {"hate_speech", "violence", "illegal_content", "adult_content"}
MODERATE = {"harassment", "misinformation"}
MINOR    = {"spam", "suspicious_link", "engagement_bait"}

ALL_VIOLATIONS = list(SEVERE | MODERATE | MINOR)

# ── Task step descriptions ────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    1: (
        "STEP 1 — VIOLATION DETECTION\n"
        "Analyze the content and detect all policy violations present.\n"
        "Violation types: hate_speech, violence, harassment, adult_content,\n"
        "misinformation, spam, suspicious_link, engagement_bait, illegal_content\n\n"
        'Action format: {"type": "detect_violations", "data": ["violation_1", ...]}'
    ),
    2: (
        "STEP 2 — COMPLIANCE SCORING\n"
        "Based on detected violations, assign a compliance score from 0.0 to 1.0.\n"
        "1.0 = fully clean, 0.0 = severe violations.\n\n"
        "Scoring guidelines:\n"
        "- No violations → 0.8 to 1.0\n"
        "- Minor violations (spam, engagement_bait) → 0.4 to 0.7\n"
        "- Moderate violations (harassment, misinformation) → 0.2 to 0.5\n"
        "- Severe violations (hate_speech, violence) → 0.0 to 0.3\n\n"
        'Action format: {"type": "score_compliance", "data": <float 0.0-1.0>}'
    ),
    3: (
        "STEP 3 — MODERATION DECISION\n"
        "Using violations and compliance score, choose the correct action.\n\n"
        "Decision rules:\n"
        "- approve: content is clean (score >= 0.7, no violations)\n"
        "- reject:  severe violations present (hate_speech, violence, illegal_content)\n"
        "         OR multiple violations with score < 0.2\n"
        "- edit:    minor/fixable violations only (spam, suspicious_link, harassment, misinformation)\n\n"
        'Action format: {"type": "moderate", "data": "approve" | "reject" | "edit"}'
    ),
    4: (
        "STEP 4 — EDIT & RESUBMIT (only if Step 3 was edit) or CONFIRM DECISION\n"
        "If you chose edit: rewrite the content to remove violations while preserving the core message.\n"
        "If you chose approve/reject: confirm your decision.\n\n"
        "Edit guidelines:\n"
        "- Remove spam phrases, suspicious links, and manipulative language\n"
        "- Soften harsh or harassing language\n"
        "- Keep the original intent intact\n\n"
        'Action format: {"type": "submit_edit", "data": "<rewritten content>"}\n'
        '           or: {"type": "confirm_approve"} / {"type": "confirm_reject"}'
    ),
}

# ── Grading functions ─────────────────────────────────────────────────────────

def _f1_score(predicted: List[str], ground_truth: List[str]) -> float:
    """Compute F1 score between predicted and ground truth violation lists."""
    if not ground_truth and not predicted:
        return 1.0
    if not ground_truth:
        return 0.0  # false positives only
    if not predicted:
        return 0.0  # missed all

    pred_set = set(predicted)
    gt_set   = set(ground_truth)

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


def _grade_detection(predicted: List[str], gt: dict) -> float:
    """Grade step 1: violation detection using F1 score."""
    f1 = _f1_score(predicted, gt["violations"])
    weight = gt["reward_weights"]["detect"]
    return round(max(0.01, min(0.99, f1 * weight * 4)), 3)  # scale to ~0-0.99


def _grade_scoring(predicted_score: float, gt: dict) -> float:
    """Grade step 2: compliance score within tolerance of ground truth."""
    true_score = gt["compliance_score"]
    error = abs(predicted_score - true_score)
    # Full credit within 0.1, partial credit up to 0.3, zero beyond
    if error <= 0.10:
        accuracy = 1.0
    elif error <= 0.20:
        accuracy = 0.7
    elif error <= 0.30:
        accuracy = 0.4
    else:
        accuracy = 0.1
    weight = gt["reward_weights"]["score"]
    return round(max(0.01, min(0.99, accuracy * weight * 4)), 3)


def _grade_decision(action: str, gt: dict, violations: List[str]) -> float:
    """Grade step 3: moderation decision correctness."""
    correct = gt["correct_action"]
    weight  = gt["reward_weights"]["decide"]

    if action == correct:
        base = 1.0
    elif action == "edit" and correct in ("reject",):
        base = 0.5   # edit is reasonable when reject is expected
    elif action == "reject" and correct == "edit":
        base = 0.4   # reject is overly strict but not wrong
    else:
        # Wrong approve on violating content is worst
        severe_present = any(v in SEVERE for v in violations)
        base = 0.0 if (action == "approve" and severe_present) else 0.15

    return round(max(0.01, min(0.99, base * weight * 4)), 3)


def _grade_confirm(action: str, gt: dict, violations_after: List[str]) -> float:
    """Grade step 4: confirmation or edit quality."""
    correct = gt["correct_action"]
    weight  = gt["reward_weights"]["confirm"]

    # If edit was done and content is now clean
    if action in ("approve", "confirm_approve"):
        base = 1.0 if not violations_after else 0.2
    elif action in ("reject", "confirm_reject"):
        base = 1.0 if violations_after or correct == "reject" else 0.2
    elif action == "submit_edit":
        # Partial credit for attempting edit
        base = 0.7 if not violations_after else 0.4
    else:
        base = 0.3

    return round(max(0.01, min(0.99, base * weight * 4)), 3)


# ── Environment ───────────────────────────────────────────────────────────────

class ContentComplianceEnvironment(Environment):
    """
    Content Compliance RL Environment — OpenEnv compliant.

    Loads 50 real content samples from data/content_samples.json.
    Grades against deterministic ground truth from data/ground_truth.json.

    4-step workflow per episode:
      Step 1: Detect violations  (graded by F1 vs ground truth)
      Step 2: Score compliance   (graded by proximity to ground truth score)
      Step 3: Decide action      (graded by correctness)
      Step 4: Confirm / Edit     (graded by final content state)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, difficulty: str = "mixed"):
        super().__init__()
        self.difficulty = difficulty
        self._state = ContentState(episode_id=None, step_count=0)
        self._sample_id: str = ""
        self._content: str = ""
        self._gt: dict = {}
        self._violations: List[str] = []
        self._score: float = 0.5
        self._step: int = 0
        self._rewards: List[float] = []
        self._action_history: List[str] = []
        self._step3_action: str = ""
        self._detected_violations: List[str] = []

    def _pick_sample(self) -> dict:
        if self.difficulty == "mixed":
            diff = random.choice(["easy", "medium", "hard"])
        else:
            diff = self.difficulty
        return random.choice(SAMPLES[diff])

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ContentObservation:
        self._reset_rubric()
        if seed is not None:
            random.seed(seed)

        sample = self._pick_sample()
        self._sample_id = sample["id"]
        self._content   = sample["content"]
        self._gt        = GROUND_TRUTH[self._sample_id]
        self._violations = list(self._gt["violations"])
        self._score      = self._gt["compliance_score"]
        self._step       = 0
        self._rewards    = []
        self._action_history  = []
        self._step3_action    = ""
        self._detected_violations = []

        self._state = ContentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            difficulty=self._gt["difficulty"],
            initial_violations=list(self._violations),
            initial_score=self._score,
            action_history=[],
        )

        return ContentObservation(
            content=self._content,
            violations=[],          # agent must detect these in step 1
            compliance_score=0.5,   # agent must score in step 2
            stage="classification",
            step_count=0,
            task_description=TASK_DESCRIPTIONS[1],
            feedback=f"New episode: {self._gt['difficulty']} difficulty. Begin with violation detection.",
            done=False,
            reward=None,
        )

    def step(self, action: ContentAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ContentObservation:
        self._step += 1
        self._state.step_count = self._step
        act = action.action_type.lower().strip()
        self._action_history.append(act)
        self._state.action_history = list(self._action_history)

        workflow_step = min(self._step, 4)

        if workflow_step == 1:
            # Step 1: violation detection — agent passes detected violations via metadata
            detected = action.metadata.get("violations", [])
            if isinstance(detected, str):
                detected = [detected]
            self._detected_violations = [v for v in detected if v in ALL_VIOLATIONS]

            reward = _grade_detection(self._detected_violations, self._gt)
            next_task = TASK_DESCRIPTIONS[2]
            stage = "classification"
            done = False
            feedback = (
                f"You detected: {self._detected_violations or 'none'}. "
                f"Ground truth has {len(self._violations)} violation(s). "
                f"Now score compliance."
            )

        elif workflow_step == 2:
            # Step 2: compliance scoring — agent passes score via metadata
            predicted_score = float(action.metadata.get("score", 0.5))
            predicted_score = max(0.0, min(1.0, predicted_score))

            reward = _grade_scoring(predicted_score, self._gt)
            next_task = TASK_DESCRIPTIONS[3]
            stage = "decision"
            done = False
            feedback = (
                f"You scored: {predicted_score:.2f}. "
                f"Ground truth score: {self._score:.2f}. "
                f"Now make moderation decision."
            )

        elif workflow_step == 3:
            # Step 3: moderation decision
            self._step3_action = act
            reward = _grade_decision(act, self._gt, self._violations)
            next_task = TASK_DESCRIPTIONS[4]
            done = False
            feedback = f"Decision: {act}. "

            if act == "edit":
                stage = "refinement"
                feedback += "Rewrite the content to fix violations in step 4."
            else:
                stage = "decision"
                feedback += f"Confirm your decision in step 4."

        else:
            # Step 4: edit resubmit or confirm
            if act == "submit_edit" and action.edited_content:
                # Re-evaluate edited content against patterns
                edited = action.edited_content
                self._content = edited
                # Check if violations are removed from edited content
                c = edited.lower()
                remaining = [
                    v for v in self._violations
                    if any(kw in c for kw in _VIOLATION_PATTERNS.get(v, []))
                ]
                reward = _grade_confirm("submit_edit", self._gt, remaining)
                feedback = (
                    f"Edit submitted. Remaining violations: {remaining or 'none'}. "
                    f"{'Content is now clean.' if not remaining else 'Some violations remain.'}"
                )
            else:
                reward = _grade_confirm(act, self._gt, self._violations)
                feedback = f"Decision confirmed: {act}."

            stage = "decision"
            done = True
            next_task = ""

        self._rewards.append(reward)

        return ContentObservation(
            content=self._content,
            violations=self._violations,
            compliance_score=self._score,
            stage=stage,
            step_count=self._step,
            task_description=next_task,
            feedback=feedback,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> ContentState:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ContentComplianceEnv",
            description=(
                "RL environment for training AI agents on content moderation. "
                "50 real content samples, deterministic grading via ground truth."
            ),
            version="1.0.0",
            author="Content Compliance RL Team",
        )


# ── Violation patterns for edit re-evaluation ─────────────────────────────────

_VIOLATION_PATTERNS = {
    "hate_speech":    ["hate", "should all die", "kill all", "i hate those"],
    "violence":       ["kill", "attack", "hurt", "weapon"],
    "harassment":     ["idiot", "stupid", "pathetic", "worthless", "ignorant", "delusional"],
    "misinformation": ["fake news", "conspiracy", "they don't want you to know", "wake up sheeple"],
    "spam":           ["click here", "buy now", "!!!", "limited time", "free prizes", "free tips"],
    "suspicious_link":["bit.ly", "tinyurl", "linktr.ee"],
    "engagement_bait":["tag someone", "share if", "🔥", "💰"],
    "adult_content":  ["explicit", "nsfw", "porn"],
    "illegal_content":["illegal", "drugs"],
}
