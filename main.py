"""Serve the customer lab API and static dashboard."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from .routes import router

app = FastAPI(title="Customer ML Lab", description="Beginner-friendly marketing analytics lab")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api")

@app.get("/api/{unmatched_path:path}")
def unknown_api_route(unmatched_path: str):
    """Keep unknown API URLs as JSON errors instead of serving the HTML page."""
    return JSONResponse(
        status_code=404,
        content={"detail": f"API route '/api/{unmatched_path}' was not found."},
    )

frontend = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
