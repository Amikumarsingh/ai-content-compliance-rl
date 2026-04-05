from fastapi import FastAPI
from app.env import ContentComplianceEnv
from tasks.task_1_easy_decision import EasyTask
from app.models import Action

app = FastAPI()
env = ContentComplianceEnv(EasyTask())

@app.post("/reset")
async def reset():
    obs = await env.reset()
    return obs.dict()

@app.post("/step")
async def step(action: dict = {}):
    try:
        act = Action(**action)
    except:
        act = Action(action_type="approve")

    obs, reward, done, _ = await env.step(act)

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done
    }

# ✅ REQUIRED FOR OPENENV
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

# ✅ ENTRY POINT
if __name__ == "__main__":
    main()
