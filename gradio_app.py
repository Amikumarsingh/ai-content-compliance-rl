"""
Gradio UI for AI Content Compliance RL Environment.
Runs directly in Hugging Face Spaces.
"""

import gradio as gr
from graders.violation_detector import ViolationDetector
from app.env import ContentComplianceEnv
from app.models import Action
import asyncio
import os

# Initialize environment
env = ContentComplianceEnv(
    max_steps=5,
    difficulty="mixed",
    evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "mock"),
)

VIOLATION_ICONS = {
    "hate_speech": "🔥 Hate Speech",
    "harassment": "😠 Harassment",
    "violence": "⚔️ Violence",
    "adult_content": "🔞 Adult Content",
    "misinformation": "⚠️ Misinformation",
    "spam": "💰 Spam",
    "suspicious_link": "🔗 Suspicious Link",
    "engagement_bait": "🎣 Engagement Bait",
    "illegal_content": "🚨 Illegal Content",
}

SAMPLES = {
    "✅ Clean Content": "Welcome to our community! We're excited to have you here. Please read our guidelines.",
    "🔥 Hate Speech": "I hate those people, they should all die and disappear forever.",
    "💰 Spam": "BUY NOW!!! Limited time offer! Click here to win $1000 FREE prizes!!!",
    "⚠️ Misinformation": "Wake up! The truth they don't want you to know about vaccines. It's all a hoax!",
    "😠 Harassment": "You're such an idiot and pathetic person. Everyone thinks you're worthless.",
    "🚨 Illegal Content": "I know where to get illegal drugs and weapons cheap, dm me.",
}


def evaluate(content: str, action: str) -> tuple:
    """Evaluate content and return results for Gradio outputs."""
    if not content.strip():
        return "⚠️ Please enter some content.", "", "N/A", "N/A", ""

    # Detect violations
    violations = ViolationDetector.detect(content)
    is_compliant = len(violations) == 0

    # Compliance score
    score = 0.9 if is_compliant else max(0.1, 1.0 - len(violations) * 0.2)

    # Determine correct action
    correct_action = "approve" if is_compliant else "reject"
    action_correct = action == correct_action

    # Reward
    if action == "approve":
        reward = 1.0 if is_compliant else 0.0
    elif action == "reject":
        reward = 1.0 if not is_compliant else 0.0
    else:  # edit
        reward = 0.5 if violations else 0.2

    # Build violation display
    if violations:
        viol_text = "\n".join(f"• {VIOLATION_ICONS.get(v, v)}" for v in violations)
    else:
        viol_text = "✅ No violations detected"

    # Result summary
    action_emoji = {"approve": "✅", "reject": "❌", "edit": "✏️"}
    result = f"{action_emoji.get(action, '')} **{action.upper()}**"
    if action_correct:
        result += " — Correct decision! 🎯"
    else:
        result += f" — Incorrect. Expected: **{correct_action}**"

    score_display = f"{score:.0%}"
    reward_display = f"{reward:.2f} / 1.00"

    return result, viol_text, score_display, reward_display, f"Expected action: **{correct_action}**"


def load_sample(sample_name: str) -> str:
    return SAMPLES.get(sample_name, "")


# ── Build Gradio UI ───────────────────────────────────────────────────────────
with gr.Blocks(
    title="🛡️ AI Content Compliance RL",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
) as demo:

    gr.Markdown("""
    # 🛡️ AI Content Compliance RL Environment
    **Train AI agents to moderate content using Reinforcement Learning**
    
    Enter content below, choose an action, and see how the RL environment evaluates it.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            content_input = gr.Textbox(
                label="📝 Content to Evaluate",
                placeholder="Enter text content here...",
                lines=4,
            )

            with gr.Row():
                action_input = gr.Radio(
                    choices=["approve", "reject", "edit"],
                    value="approve",
                    label="🎮 Action",
                )
                evaluate_btn = gr.Button("🚀 Evaluate", variant="primary", scale=1)

            gr.Markdown("### 🎯 Sample Content")
            with gr.Row():
                for name in SAMPLES:
                    gr.Button(name, size="sm").click(
                        fn=lambda n=name: load_sample(n),
                        outputs=content_input,
                    )

        with gr.Column(scale=2):
            result_output = gr.Markdown(label="Result")

            with gr.Row():
                score_output = gr.Label(label="📊 Compliance Score")
                reward_output = gr.Label(label="🏆 Reward")

            violations_output = gr.Textbox(
                label="⚠️ Violations Detected",
                lines=4,
                interactive=False,
            )
            suggestion_output = gr.Markdown()

    evaluate_btn.click(
        fn=evaluate,
        inputs=[content_input, action_input],
        outputs=[result_output, violations_output, score_output, reward_output, suggestion_output],
    )

    gr.Markdown("""
    ---
    ### 📋 How It Works
    | Action | When to use | Reward |
    |--------|-------------|--------|
    | ✅ Approve | Content is clean and compliant | +1.0 if correct |
    | ❌ Reject | Content violates policies | +1.0 if correct |
    | ✏️ Edit | Content needs modification | +0.5 |
    
    **Violation Types:** Hate Speech • Harassment • Violence • Adult Content • Misinformation • Spam • Illegal Content
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
