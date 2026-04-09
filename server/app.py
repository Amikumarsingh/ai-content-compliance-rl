"""Server entry point — delegates to hf_server."""
import os
import uvicorn
from hf_server import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()
