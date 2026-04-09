"""
CLI Entry Point for Content Compliance RL.

Usage:
    python main.py demo          Run demo inference
    python main.py train         Train an agent
    python main.py serve         Start the API server
    python main.py test          Run tests
"""

import argparse
import sys


def run_demo():
    """Run demo inference."""
    from inference import main as inference_main
    import asyncio
    asyncio.run(inference_main())


def run_train():
    """Run training."""
    print("Training mode...")
    try:
        from agents.q_agent import QLearningAgent  # type: ignore
        from openenv import ContentComplianceEnv, Action  # type: ignore
    except ImportError as e:
        print(f"Training dependencies not available: {e}")
        return
    import asyncio

    async def train():
        env = ContentComplianceEnv(max_steps=5, difficulty="mixed")
        agent = QLearningAgent(actions=["approve", "reject", "edit"])

        episodes = 100
        print(f"Training for {episodes} episodes...")

        for ep in range(1, episodes + 1):
            obs = await env.reset()
            total_reward = 0.0
            steps = 0

            while True:
                state = obs.content[:100]
                action = agent.select_action(state)
                action_obj = Action(action_type=action)

                result = await env.step(action_obj)
                agent.store_experience(
                    state, action, result.reward.value,
                    result.observation.content[:100], result.done
                )

                total_reward += result.reward.value
                steps += 1

                if result.done:
                    break

                obs = result.observation

            agent.train_batch(batch_size=32)
            agent.record_episode(ep, total_reward, steps, total_reward > 0.5)

            if ep % 10 == 0:
                stats = agent.get_statistics()
                print(f"Episode {ep}: avg_reward={stats['avg_reward']:.3f}, "
                      f"success_rate={stats['success_rate']:.2f}, epsilon={agent.epsilon:.3f}")

        print("Training complete!")
        return agent

    asyncio.run(train())


def run_serve():
    """Start the API server."""
    import uvicorn
    from hf_server import app  # type: ignore

    port = 7860
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_tests():
    """Run tests."""
    try:
        import pytest  # type: ignore
    except ImportError:
        print("pytest not installed. Run: pip install pytest")
        sys.exit(1)
    sys.exit(pytest.main(["-v", "graders/", "rewards/"]))


def main():
    parser = argparse.ArgumentParser(description="Content Compliance RL")
    parser.add_argument(
        "command",
        choices=["demo", "train", "serve", "test"],
        help="Command to run",
    )

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "train":
        run_train()
    elif args.command == "serve":
        run_serve()
    elif args.command == "test":
        run_tests()


if __name__ == "__main__":
    main()
