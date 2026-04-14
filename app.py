from __future__ import annotations

import uvicorn

from pyrad_workflow.app import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8181, reload=True)
