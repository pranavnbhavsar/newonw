
# Improved bridge to run the project's fetcher and stream output to a log file used by the UI.
# Features:
# - Auto-detects fetcher.py by searching upward from the bridge location and inside the project tree.
# - Uses text mode for subprocess to avoid the RuntimeWarning and allow line-buffered reading.
# - Works on Windows and Unix without absolute sandbox paths.
# - If multiple fetcher-like files exist, prefers a file named exactly 'fetcher.py' in the project root.
import subprocess, sys, os, time, shlex

LOG = os.path.join(os.path.dirname(__file__), 'predictions.log')

def find_fetcher():
    # Search upward from the bridge directory for a fetcher.py or similar
    start = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # First look for fetcher.py at start and its immediate children
    candidates = []
    for root, dirs, files in os.walk(start):
        for f in files:
            if f.lower() == 'fetcher.py':
                return os.path.join(root, f)
            if 'fetcher' in f.lower() and f.lower().endswith('.py'):
                candidates.append(os.path.join(root, f))
    # fallback to any candidate found
    if candidates:
        return candidates[0]
    # As last resort, look one directory above start
    parent = os.path.abspath(os.path.join(start, '..'))
    for root, dirs, files in os.walk(parent):
        for f in files:
            if 'fetcher' in f.lower() and f.lower().endswith('.py'):
                return os.path.join(root, f)
    return None

def main():
    FETCHER = find_fetcher()
    if not FETCHER:
        print('ERROR: Could not locate fetcher.py. Place fetcher.py in the project root or in a subfolder.')
        print('Searched starting from:', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        return 1

    # Use the same Python interpreter running this bridge
    PY = sys.executable or 'python'
    cmd = [PY, '-u', FETCHER]
    print('Starting fetcher bridge: ' + ' '.join(shlex.quote(c) for c in cmd))
    # Ensure log exists
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8', buffering=1) as lf:
        lf.write('\\n=== Bridge started at ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ===\\n')
        # Use text mode so we can iterate line by line without binary buffering warnings
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in p.stdout:
                if not line:
                    break
                # print to bridge stdout and append to log
                print(line, end='')
                lf.write(line)
        except KeyboardInterrupt:
            print('\\nBridge interrupted by user.')
        finally:
            try:
                p.terminate()
            except Exception:
                pass
            lf.write('\\n=== Bridge stopped at ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ===\\n')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
