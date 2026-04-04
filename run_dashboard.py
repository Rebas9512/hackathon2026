#!/usr/bin/env python3
"""
Spillover Alpha — Dashboard Launcher

Usage:
    python3 run_dashboard.py          # default port 8501
    python3 run_dashboard.py --port 8888
"""

import subprocess
import sys
import os
import webbrowser
import threading
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent / "dashboard"
APP_FILE = DASHBOARD_DIR / "app.py"


def check_dependencies():
    """Check that required packages are installed."""
    missing = []
    for pkg in ["streamlit", "plotly", "pandas", "numpy", "sklearn", "xgboost", "networkx"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("sklearn", "scikit-learn"))
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip3 install {' '.join(missing)}")
        sys.exit(1)


def main():
    check_dependencies()

    port = "8501"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]

    print(f"""
    ╔═══════════════════════════════════════════════╗
    ║           ⚡  Spillover Alpha                 ║
    ║   Sentiment Spillover Network × Earnings      ║
    ║   2026 FinHack Challenge — Case 4             ║
    ╚═══════════════════════════════════════════════╝

    Starting dashboard on port {port}...
    Open: http://localhost:{port}
    """)

    os.chdir(DASHBOARD_DIR)

    url = f"http://localhost:{port}"
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(APP_FILE),
        "--server.port", port,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.primaryColor", "#A855F7",
        "--theme.backgroundColor", "#0A0A0A",
        "--theme.secondaryBackgroundColor", "#111114",
        "--theme.textColor", "#FFFFFF",
    ])


if __name__ == "__main__":
    main()
