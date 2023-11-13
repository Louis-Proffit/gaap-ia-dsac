from flask import Flask, render_template, send_from_directory, request
from model import get_embeddings, get_similar_documents, get_similar_chunks, get_db, get_pipeline, get_conversation

app = Flask(__name__)

# db = get_db("./embeddings", get_embeddings())
pipeline = get_pipeline()
conversation = get_conversation()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global conversation
    query = request.args.get('msg')
    conversation.add_message({"role": "user", "content": query})
    conversation = pipeline(conversation)
    new_message = conversation.generated_responses[-1]
    conversation.add_message({"role": "assistant", "content": new_message})
    return new_message


@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.debug = True
    app.run()
