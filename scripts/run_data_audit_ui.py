from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PORT = "8501"


def ensure_project_imports() -> None:
    src = str(SRC_DIR)
    current = os.environ.get("PYTHONPATH", "")
    if src not in current.split(os.pathsep):
        os.environ["PYTHONPATH"] = src if not current else f"{src}{os.pathsep}{current}"
    if src not in sys.path:
        sys.path.insert(0, src)


ensure_project_imports()


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        return False
    return get_script_run_ctx() is not None


def launch_streamlit() -> None:
    script_path = Path(__file__).resolve()
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    python = venv_python if venv_python.exists() else Path(sys.executable)
    args = sys.argv[1:]
    if "--server.port" not in args:
        args = [*args, "--server.port", DEFAULT_PORT]
    port = args[args.index("--server.port") + 1] if "--server.port" in args else DEFAULT_PORT
    print(f"Starting Streamlit data audit UI: http://localhost:{port}", flush=True)
    cmd = [
        str(python),
        "-m",
        "streamlit",
        "run",
        str(script_path),
        *args,
    ]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy(), check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    if running_inside_streamlit():
        from match_dynamics.audit_ui import main

        main()
    else:
        launch_streamlit()
