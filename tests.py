# Use a pipeline as a high-level helper
from transformers import pipeline, Conversation


if __name__ == "__main__":
    pipe = pipeline("conversational", model="facebook/blenderbot-1B-distill")

    conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
    conversation_1.add_message(
        {"role": "assistant", "content": "What kind of movie are you going to see? I love going to the theatre."})
    conversation_1.add_message({"role": "user", "content": "I would like to go to see an action movie"})

    x = pipe(conversation_1)

    print(x)