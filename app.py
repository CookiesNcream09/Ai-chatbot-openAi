from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()


# Create an OpenAIChat instance with GPT-3.5-turbo
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get("OPENAI_API_KEY"))

app = Flask(__name__)

chat_messages = []
chat_messages.append(SystemMessage(content="You are a helpful assistant."))

@app.route("/chat_message", methods=["POST"])
def gpt3_chat():
    # Convert the JSON request body to a dictionary
    request_data = request.get_json()

    # Check if the message parameter is present
    if "message" not in request_data:
        return jsonify({"error": "Message parameter is missing"}), 400

    chat_messages.append(HumanMessage(content=request_data["message"]))


    # Generate a response from GPT-3.5-turbo using LangChain
    response = chat_model(chat_messages).content

    chat_messages.append(AIMessage(content=response))

    return jsonify({"assistant": response["assistant"]})

@app.route("/chat_message", methods=["GET"])
def list_chat():
    return chat_messages


if __name__ == "_main_":
    app.run(debug=True)
