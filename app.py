from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
import uvicorn


from backend import get_runtime_info, run_workflow


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Self-Correcting Multi-Agent App")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")



class RunRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=300)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "example_topic": "What is an AI agent?",
            "runtime": get_runtime_info(),
        },
    )


@app.get("/api/config")
def config():
    """Safe runtime information for the UI/demo. Never returns secrets."""
    return get_runtime_info()


@app.post("/api/run")
def run_agents(payload: RunRequest):
    topic = payload.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Please enter a topic.")

    try:
        return run_workflow(topic)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc





# if __name__ == "__main__":
#     # DigitalOcean App Platform provides PORT automatically.
#     port = int(__import__("os").getenv("PORT", "8000"))
#     uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
