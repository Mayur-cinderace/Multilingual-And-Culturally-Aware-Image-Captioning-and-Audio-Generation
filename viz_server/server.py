# viz_server/server.py
from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "index.html")

@app.route("/static/<path:filename>")
def serve_static(filename):
    static_dir = os.path.join(SCRIPT_DIR, "static")
    return send_from_directory(static_dir, filename)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Cultural AI Visualization Server")
    print("=" * 60)
    print("\n📊 Server starting on http://localhost:8081")
    print("📁 Serving from:", SCRIPT_DIR)
    print("\n💡 Make sure you've run 'run_cognition_viz.py' first!")
    print("   This generates the visualization HTML files.\n")
    print("=" * 60)
    print()
    
    app.run(host="0.0.0.0", port=8081, debug=True)