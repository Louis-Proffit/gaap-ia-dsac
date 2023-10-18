from flask import Flask, render_template, send_from_directory

app = Flask(__name__)
app.static_folder = "static"

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    return render_template("index.html")

@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.run()