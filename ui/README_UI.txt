
MrPerfectV5 UI additions (fixed)
-----------------------
What was added/updated:
- ui/app.py        : small Flask app serving the UI at http://localhost:8000/
- ui/templates/index.html : main page that polls and displays the latest lines from predictions.log
- ui/fetcher_ui_bridge.py : runs the project's fetcher and streams stdout to ui/predictions.log (auto-detecting)
- ui/predictions.log : example log file (created empty)

How to use (on your machine):
1) Install dependencies (prefer a venv):
   pip install flask

2) From the project root (the directory that contains your project files), start the bridge to run fetcher and write to the log:
   python ui/fetcher_ui_bridge.py

   This bridge will auto-detect fetcher.py in the project tree. It uses the same python executable you're using for the bridge,
   so if you want a specific Python, run the bridge with that python (e.g., C:\\path\\to\\python.exe ui\\fetcher_ui_bridge.py).

3) Start the UI (in a separate terminal):
   python ui/app.py

   Then open http://localhost:8000 in your browser to see live output.

Notes:
- If your fetcher requires environment variables or config files, make sure to set them before starting the bridge.
- If multiple fetcher-like scripts exist, the bridge prefers a file named exactly 'fetcher.py' found in the project tree.
