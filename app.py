from flask import Flask, jsonify, request
from pydantic_agent import run_agent
from qa_dict import qa_dict
import logfire

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")

app = Flask(__name__)

# Sample data
# items = [
#     {"id": 1, "name": "Item 1"},
#     {"id": 2, "name": "Item 2"},
#     {"id": 3, "name": "Item 3"}
# ]

# # Route to get all items
# @app.route('/api/items', methods=['GET'])
# def get_items():
#     return jsonify(items)

# # Route to get a single item by ID
# @app.route('/api/items/<int:item_id>', methods=['GET'])
# def get_item(item_id):
#     item = next((item for item in items if item['id'] == item_id), None)
#     if item:
#         return jsonify(item)
#     else:
#         return jsonify({"message": "Item not found"}), 404

# # Route to create a new item
# @app.route('/api/items', methods=['POST'])
# def create_item():
#     new_item = request.get_json()
#     items.append(new_item)
#     return jsonify(new_item), 201


# {
#     "message": "Why do we have to pray?",
#     "first_name": "Ahmad",
#     "last_name": "Soliman",
#     "user_id": "412",
#     "message_id": "124",
#     "chat_id": "123"
# }
# IslamQA AI Chatbot API Endpoint
@app.route("/api/chat", methods=["POST"])
async def chat():
    msg_request = request.get_json()
    user_input = msg_request["message"]
    result = await run_agent(user_input)
    return jsonify(result), 200


@app.route("/api/report", methods=["POST"])
def report():
    return "Reported", 200


if __name__ == "__main__":
    app.run(debug=True)
