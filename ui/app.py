from flask import Flask, render_template, jsonify, send_from_directory
import os, time

app = Flask(__name__, template_folder='templates', static_folder='static')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'predictions.log')

def tail_lines(filename, n=200):
    try:
        with open(filename, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 1024
            data = b''
            while size > 0 and data.count(b'\n') <= n:
                if size - block > 0:
                    f.seek(size - block)
                    chunk = f.read(block)
                else:
                    f.seek(0)
                    chunk = f.read(size)
                data = chunk + data
                size -= block
            lines = data.splitlines()[-n:]
            return [line.decode('utf-8', errors='replace') for line in lines]
    except FileNotFoundError:
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/log')
def log():
    lines = tail_lines(LOG_FILE, 500)
    return jsonify(lines)

@app.route('/download/<path:filename>')
def download_file(filename):
    # allow downloading any file from project root safely
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.abspath(os.path.join(base, filename))
    if not path.startswith(base):
        return "Invalid path", 400
    directory = os.path.dirname(path)
    return send_from_directory(directory, os.path.basename(path), as_attachment=True)

if __name__ == '__main__':
    # Run on all interfaces so you can access from your browser on localhost
    app.run(host='0.0.0.0', port=8000, debug=False)
