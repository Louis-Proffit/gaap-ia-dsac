from flask import Flask, render_template, send_from_directory, request
from model import get_embeddings, get_similar_documents, get_similar_chunks, get_db

app = Flask(__name__)

db = get_db("./emb", get_embeddings())

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    documents = get_similar_documents(query, db)
    chunks = get_similar_chunks(documents)
    return list(chunks)


@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.debug = True
    app.run()
