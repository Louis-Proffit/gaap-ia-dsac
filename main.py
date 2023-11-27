from flask import Flask, render_template, send_from_directory, request
from model import get_embeddings, get_similar_documents, get_db
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

db = get_db("./embeddings", get_embeddings())
client = OpenAI()
messages = []

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    chunks = get_similar_documents(query, db)
    context = ("Utilise les éléments de contexte suivants pour répondre à la question à la fin. Utilise cinq phrases "
               "au maximum.\n")
    for document in chunks:
        context += f"- {document.page_content}\n"
    context += f"Question: {query}"
    print(context)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": context}]
    )
    return completion.choices[0].message.content


@app.route('/static/<path:path>')
def send_report(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.debug = True
    app.run()
