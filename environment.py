"""
Content Compliance RL Environment - OpenEnv compliant implementation.

Loads 50 real content samples from data/content_samples.json.
Grades against deterministic ground truth from data/ground_truth.json.

4-step workflow with difficulty-aware grading:
  Step 1: Detect violations  - partial credit per violation (F1 score)
  Step 2: Score compliance   - proximity to ground truth score
  Step 3: Decide action      - correctness with partial credit
  Step 4: Confirm / Edit     - final content state evaluation

Difficulty enforced:
  easy:   obvious violations, binary correct/wrong expected
  medium: subtle violations, edit workflow expected
  hard:   multi-violation, coded language, requires reasoning
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from models import ContentAction, ContentObservation, ContentState

# -- Load data files ----------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_samples() -> Dict[str, List[dict]]:
    path = os.path.join(_DATA_DIR, "content_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    grouped: Dict[str, List[dict]] = {"easy": [], "medium": [], "hard": []}
    for s in samples:
        grouped[s["difficulty"]].append(s)
    return grouped


def _load_ground_truth() -> Dict[str, dict]:
    path = os.path.join(_DATA_DIR, "ground_truth.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SAMPLES = _load_samples()
GROUND_TRUTH = _load_ground_truth()

# -- Violation taxonomy -------------------------------------------------------

SEVERE   = {"hate_speech", "violence", "illegal_content", "adult_content"}
MODERATE = {"harassment", "misinformation"}
MINOR    = {"spam", "suspicious_link", "engagement_bait"}
ALL_VIOLATIONS = list(SEVERE | MODERATE | MINOR)

# Precise multi-word patterns only — avoids false positives from single words
_VIOLATION_PATTERNS = {
    "hate_speech":    ["should all die", "kill all", "i hate those", "they should die"],
    "violence":       ["kill them", "attack them", "hurt people", "weapon of"],
    "harassment":     ["you are an idiot", "you are stupid", "you are pathetic",
                       "you are worthless", "too ignorant to", "so delusional"],
    "misinformation": ["fake news", "they don't want you to know", "wake up sheeple",
                       "mainstream media won't", "scientists are hiding"],
    "spam":           ["click here", "buy now", "limited time offer", "free prizes",
                       "free tips inside", "act now", "win $"],
    "suspicious_link":["bit.ly/", "tinyurl.com/", "linktr.ee/"],
    "engagement_bait":["tag someone who", "share if you", "comment below and i"],
    "adult_content":  ["explicit content", "nsfw", "dm for pics"],
    "illegal_content":["buy illegal", "sell drugs", "illegal weapons"],
}

# -- Difficulty-aware grading parameters --------------------------------------

# How strictly each difficulty is graded
_DIFFICULTY_CONFIG = {
    "easy": {
        # Easy: agent should get violations exactly right — tight tolerance
        "detection_tolerance": 0,    # must match exactly for full credit
        "score_tolerance":     0.10, # score within 0.10 of GT
        "partial_credit":      0.3,  # partial credit for close-but-wrong
        "step_weights":        {"detect": 0.30, "score": 0.20, "decide": 0.35, "confirm": 0.15},
    },
    "medium": {
        # Medium: some tolerance, edit workflow expected
        "detection_tolerance": 1,    # can miss 1 violation and still get partial
        "score_tolerance":     0.15,
        "partial_credit":      0.5,
        "step_weights":        {"detect": 0.25, "score": 0.20, "decide": 0.30, "confirm": 0.25},
    },
    "hard": {
        # Hard: coded language, multi-violation — more tolerance, reasoning rewarded
        "detection_tolerance": 2,    # can miss up to 2 violations
        "score_tolerance":     0.20,
        "partial_credit":      0.6,
        "step_weights":        {"detect": 0.25, "score": 0.15, "decide": 0.35, "confirm": 0.25},
    },
}

# -- Task descriptions --------------------------------------------------------

TASK_DESCRIPTIONS = {
    1: (
        "STEP 1 - VIOLATION DETECTION\n"
        "Analyze the content and detect all policy violations present.\n"
        "Violation types: hate_speech, violence, harassment, adult_content,\n"
        "misinformation, spam, suspicious_link, engagement_bait, illegal_content\n\n"
        "Return JSON: {\"violations\": [\"type1\", \"type2\"]} or {\"violations\": []}"
    ),
    2: (
        "STEP 2 - COMPLIANCE SCORING\n"
        "Assign a compliance score from 0.0 to 1.0 based on violations found.\n"
        "- No violations: 0.8 to 1.0\n"
        "- Minor violations (spam, engagement_bait, suspicious_link): 0.4 to 0.7\n"
        "- Moderate violations (harassment, misinformation): 0.2 to 0.5\n"
        "- Severe violations (hate_speech, violence, illegal_content): 0.0 to 0.3\n\n"
        "Return JSON: {\"score\": 0.85}"
    ),
    3: (
        "STEP 3 - MODERATION DECISION\n"
        "Choose the correct moderation action:\n"
        "- approve: content is clean (score >= 0.7, no violations)\n"
        "- reject:  severe violations (hate_speech, violence, illegal_content)\n"
        "          OR score < 0.2 with multiple violations\n"
        "- edit:    minor/fixable violations (spam, suspicious_link, harassment, misinformation)\n\n"
        "Respond with exactly one word: approve, reject, or edit"
    ),
    4: (
        "STEP 4 - EDIT & RESUBMIT or CONFIRM\n"
        "If step 3 was edit: rewrite the content removing all violations.\n"
        "If step 3 was approve/reject: confirm your decision.\n\n"
        "For edit: return the rewritten text only.\n"
        "For confirm: respond with confirm_approve or confirm_reject"
    ),
}

# -- Grading functions --------------------------------------------------------

def _f1_score(predicted: List[str], ground_truth: List[str]) -> float:
    """F1 score between predicted and ground truth violation sets."""
    if not ground_truth and not predicted:
        return 1.0
    if not ground_truth:
        # False positives only — penalise but not harshly
        return max(0.0, 1.0 - len(predicted) * 0.2)
    if not predicted:
        return 0.0
    pred_set = set(predicted)
    gt_set   = set(ground_truth)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


def _grade_detection(predicted: List[str], gt: dict, difficulty: str) -> float:
    """
    Grade step 1 violation detection.
    - Easy: tight — must match exactly for full credit
    - Medium: can miss 1 violation
    - Hard: can miss up to 2 violations, coded language is hard
    """
    cfg = _DIFFICULTY_CONFIG[difficulty]
    gt_violations = gt["violations"]
    weight = cfg["step_weights"]["detect"]

    f1 = _f1_score(predicted, gt_violations)

    # Difficulty-aware scaling
    if difficulty == "easy":
        # Binary-ish: full credit for exact match, steep penalty for misses
        scaled = f1
    elif difficulty == "medium":
        # Partial credit: missing 1 out of 2 still gets 0.5
        missed = len(set(gt_violations) - set(predicted))
        tolerance = cfg["detection_tolerance"]
        if missed <= tolerance:
            scaled = max(f1, 0.6)  # floor at 0.6 if within tolerance
        else:
            scaled = f1
    else:  # hard
        # More forgiving — hard content has coded language
        missed = len(set(gt_violations) - set(predicted))
        tolerance = cfg["detection_tolerance"]
        if missed <= tolerance:
            scaled = max(f1, 0.5)
        else:
            scaled = max(f1, 0.2)  # always some credit for trying

    return round(max(0.01, min(0.99, scaled * weight * 4)), 3)


def _grade_scoring(predicted_score: float, gt: dict, difficulty: str) -> float:
    """
    Grade step 2 compliance scoring.
    Tolerance increases with difficulty.
    """
    cfg = _DIFFICULTY_CONFIG[difficulty]
    true_score = gt["compliance_score"]
    weight = cfg["step_weights"]["score"]
    tolerance = cfg["score_tolerance"]

    error = abs(predicted_score - true_score)
    if error <= tolerance:
        accuracy = 1.0
    elif error <= tolerance * 2:
        accuracy = 0.7
    elif error <= tolerance * 3:
        accuracy = 0.4
    else:
        accuracy = 0.1

    return round(max(0.01, min(0.99, accuracy * weight * 4)), 3)


def _grade_decision(action: str, gt: dict, violations: List[str], difficulty: str) -> float:
    """
    Grade step 3 moderation decision.
    - Easy: binary correct/wrong
    - Medium/Hard: partial credit for reasonable alternatives
    """
    correct = gt["correct_action"]
    cfg = _DIFFICULTY_CONFIG[difficulty]
    weight = cfg["step_weights"]["decide"]
    partial = cfg["partial_credit"]

    if action == correct:
        base = 1.0
    elif difficulty == "easy":
        # Easy: no partial credit — should be obvious
        severe_present = any(v in SEVERE for v in violations)
        base = 0.01 if (action == "approve" and severe_present) else 0.1
    else:
        # Medium/Hard: partial credit for reasonable alternatives
        if action == "edit" and correct == "reject":
            base = partial * 0.8   # edit is reasonable when reject expected
        elif action == "reject" and correct == "edit":
            base = partial * 0.6   # reject is overly strict but not wrong
        elif action == "approve" and correct == "edit":
            base = partial * 0.3   # approve when edit needed — too lenient
        else:
            severe_present = any(v in SEVERE for v in violations)
            base = 0.01 if (action == "approve" and severe_present) else 0.15

    return round(max(0.01, min(0.99, base * weight * 4)), 3)


def _grade_confirm(action: str, gt: dict, violations_after: List[str], difficulty: str) -> float:
    """
    Grade step 4 confirmation or edit quality.
    Hard difficulty gives more credit for partial edits.
    """
    correct = gt["correct_action"]
    cfg = _DIFFICULTY_CONFIG[difficulty]
    weight = cfg["step_weights"]["confirm"]

    if action in ("approve", "confirm_approve"):
        base = 1.0 if not violations_after else 0.2
    elif action in ("reject", "confirm_reject"):
        base = 1.0 if violations_after or correct == "reject" else 0.2
    elif action == "submit_edit":
        if not violations_after:
            base = 1.0   # all violations removed
        elif difficulty == "hard":
            base = 0.6   # partial edit on hard content still gets credit
        else:
            base = 0.4
    else:
        base = 0.2

    return round(max(0.01, min(0.99, base * weight * 4)), 3)


def _default_gt(sample: dict) -> dict:
    diff = sample.get("difficulty", "easy")
    cfg = _DIFFICULTY_CONFIG[diff]
    return {
        "violations": [],
        "compliance_score": 0.9,
        "correct_action": "approve",
        "difficulty": diff,
        "reward_weights": cfg["step_weights"],
    }


# -- Environment --------------------------------------------------------------

class ContentComplianceEnvironment(Environment):
    """
    Content Compliance RL Environment - OpenEnv compliant.

    50 real content samples loaded from data/content_samples.json.
    Deterministic grading against data/ground_truth.json.
    Difficulty-aware grading: easy/medium/hard have meaningfully different
    tolerance, partial credit, and reward distributions.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, difficulty: str = "mixed"):
        super().__init__()
        self.difficulty = difficulty
        self._sample_id: str = ""
        self._content: str = ""
        self._gt: dict = {}
        self._difficulty: str = "easy"
        self._violations: List[str] = []
        self._score: float = 0.5
        self._step: int = 0
        self._rewards: List[float] = []
        self._action_history: List[str] = []
        self._detected_violations: List[str] = []
        self._step3_action: str = ""
        self._state = ContentState(episode_id=None, step_count=0)

    def _pick_sample(self, forced_difficulty: Optional[str] = None) -> dict:
        diff = forced_difficulty or (
            random.choice(["easy", "medium", "hard"])
            if self.difficulty == "mixed" else self.difficulty
        )
        return random.choice(SAMPLES[diff])

    def _init_from_sample(self, sample: dict, episode_id: Optional[str] = None) -> None:
        self._sample_id  = sample["id"]
        self._content    = sample["content"]
        self._gt         = GROUND_TRUTH.get(self._sample_id, _default_gt(sample))
        self._difficulty = self._gt.get("difficulty", sample.get("difficulty", "easy"))
        self._violations = list(self._gt["violations"])
        self._score      = self._gt["compliance_score"]
        self._step       = 0
        self._rewards    = []
        self._action_history  = []
        self._detected_violations = []
        self._step3_action    = ""
        self._state = ContentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            difficulty=self._difficulty,
            initial_violations=list(self._violations),
            initial_score=self._score,
            action_history=[],
        )

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> ContentObservation:
        self._reset_rubric()
        if seed is not None:
            random.seed(seed)
        # Respect difficulty passed via kwargs (e.g. from inference per-episode)
        forced = kwargs.get("difficulty")
        self._init_from_sample(self._pick_sample(forced), episode_id)
        return ContentObservation(
            content=self._content,
            violations=[],       # agent must detect in step 1
            compliance_score=0.5,
            stage="classification",
            step_count=0,
            task_description=TASK_DESCRIPTIONS[1],
            feedback=(
                f"New episode [{self._difficulty.upper()}]. "
                f"Content loaded. Begin with violation detection."
            ),
            done=False,
            reward=0.001,
        )

    def step(self, action: ContentAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ContentObservation:
        # Auto-init if reset() was never called (stateless HTTP mode)
        if not self._sample_id:
            self._init_from_sample(self._pick_sample())

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
            reward    = _grade_detection(self._detected_violations, self._gt, self._difficulty)
            next_task = TASK_DESCRIPTIONS[2]
            stage     = "classification"
            done      = False
            n_gt = len(self._violations)
            n_det = len(self._detected_violations)
            feedback  = (
                f"[{self._difficulty.upper()}] You detected {n_det} violation(s): "
                f"{self._detected_violations or 'none'}. "
                f"Ground truth has {n_gt}. "
                f"Reward: {reward}. Now score compliance."
            )

        elif workflow_step == 2:
            # Step 2: compliance scoring — agent passes score via metadata
            predicted_score = float(action.metadata.get("score", 0.5))
            predicted_score = max(0.0, min(1.0, predicted_score))
            reward    = _grade_scoring(predicted_score, self._gt, self._difficulty)
            next_task = TASK_DESCRIPTIONS[3]
            stage     = "decision"
            done      = False
            feedback  = (
                f"[{self._difficulty.upper()}] You scored {predicted_score:.2f}. "
                f"Ground truth: {self._score:.2f}. "
                f"Reward: {reward}. Now make moderation decision."
            )

        elif workflow_step == 3:
            # Step 3: moderation decision
            self._step3_action = act
            reward    = _grade_decision(act, self._gt, self._violations, self._difficulty)
            next_task = TASK_DESCRIPTIONS[4]
            done      = False
            if act == "edit":
                stage    = "refinement"
                feedback = (
                    f"[{self._difficulty.upper()}] Edit chosen. "
                    f"Reward: {reward}. Rewrite content in step 4."
                )
            else:
                stage    = "decision"
                feedback = (
                    f"[{self._difficulty.upper()}] Decision: {act}. "
                    f"Reward: {reward}. Confirm in step 4."
                )

        else:
            # Step 4: edit resubmit or confirm
            if act == "submit_edit" and action.edited_content:
                self._content = action.edited_content
                c = self._content.lower()
                remaining = [
                    v for v in self._violations
                    if any(kw in c for kw in _VIOLATION_PATTERNS.get(v, []))
                ]
                reward   = _grade_confirm("submit_edit", self._gt, remaining, self._difficulty)
                feedback = (
                    f"[{self._difficulty.upper()}] Edit submitted. "
                    f"Remaining violations: {remaining or 'none'}. "
                    f"{'Content is now clean.' if not remaining else 'Some violations remain.'} "
                    f"Reward: {reward}."
                )
            else:
                reward   = _grade_confirm(act, self._gt, self._violations, self._difficulty)
                feedback = (
                    f"[{self._difficulty.upper()}] Decision confirmed: {act}. "
                    f"Reward: {reward}."
                )
            stage     = "decision"
            done      = True
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
                "RL environment for content moderation. "
                "50 real samples, deterministic grading, difficulty-aware rewards."
            ),
            version="1.0.0",
            author="Content Compliance RL Team",
        )
