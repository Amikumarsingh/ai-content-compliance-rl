"""
Server entry point — uses openenv-core create_app for full spec compliance.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openenv.core.env_server.http_server import create_app
from models import ContentAction, ContentObservation
from environment import ContentComplianceEnvironment

app = create_app(
    ContentComplianceEnvironment,
    ContentAction,
    ContentObservation,
    env_name="content_compliance_env",
    max_concurrent_envs=4,
)


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
