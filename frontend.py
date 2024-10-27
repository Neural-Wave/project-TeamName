from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="static")

# Serve the main HTML file
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

# Serve other files as needed (optional if everything is in `static`)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)

if __name__ == "__main__":
    app.run(debug=True)