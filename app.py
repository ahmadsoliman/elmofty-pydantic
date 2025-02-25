from flask import Flask, jsonify, request
from pydantic_agent import pydantic_islam_agent, PydanticAIDeps, RAGToolTracker
from pydantic_ai.usage import UsageLimits
from qa_dict import qa_dict

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


async def run_agent(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps()

    # Run the agent in a stream
    result = await pydantic_islam_agent.run(
        user_input,
        deps=deps,
        # message_history=st.session_state.messages[:-1],
        usage_limits=UsageLimits(request_limit=6),
    )

    response = result.data.response
    source_questions_ids = result.data.source_questions_ids

    similar_qas = [
        "**سؤال([{0}](https://islamqa.info/ar/answers/{0}/)):** {1} \\\n\\\n**الإجابة:** {2}".format(
            qa_id,
            qa_dict[qa_id].question.replace("\n", "\\\n"),
            qa_dict[qa_id].answer.replace("\n", "\\\n"),
        )
        for qa_id in source_questions_ids
        if qa_id in qa_dict
    ]
    formatted_similar_qas = ""
    if similar_qas and len(similar_qas) > 0:
        # format the similar questions and answers
        formatted_similar_qas = "#### الأسئلة المشابهة: \n" + "\\\n \\\n".join(
            similar_qas
        )

    return {
        "response": response,
        "source_questions_ids": source_questions_ids,
        "similar_qas": similar_qas,
        "formatted_similar_qas": formatted_similar_qas,
        "message": response + "\\\n\\\n" + formatted_similar_qas,
    }


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
async def create_item():
    msg_request = request.get_json()
    user_input = msg_request["message"]
    result = await run_agent(user_input)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)
