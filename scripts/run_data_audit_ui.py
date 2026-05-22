from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from match_dynamics.audit_ui import main


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        return False
    return get_script_run_ctx() is not None


def launch_streamlit() -> None:
    script_path = Path(__file__).resolve()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        launch_streamlit()
