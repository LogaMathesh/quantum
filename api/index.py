from pathlib import Path

from backend.main import app


FRONTEND_BUILD = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if not FRONTEND_BUILD.is_dir():
    raise FileNotFoundError(
        f"Frontend build not found at {FRONTEND_BUILD}. "
        "Build the Vite frontend before starting the Vercel application."
    )

app.frontend(
    "/",
    directory=FRONTEND_BUILD,
    fallback="index.html",
)