from flask import Flask, render_template, send_from_directory, request
from index import load_query_engine
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

query_engine = load_query_engine()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    return query_engine.query(query).response


@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.debug = True
    app.run()
