# In the route handlers, perform the RAG process to retrieve
# suggestions from MongoDB based on the user's code submission

from flask import Flask, request, jsonify
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json

import subprocess

app = Flask(__name__)


# MongoDB setup
client = MongoClient('mongodb+srv://vickyye:R7GMsZDqY1vRowQb@leetcodeans.uh2nk.mongodb.net/?retryWrites=true&w=majority&appName=leetcodeans')
db = client['leetcode']
collection = db['solutions']

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example test cases
test_cases = [
    {
        "input": [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]],
        "expected": [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
    }
]

# When the user submits code check against the test cases
@app.route('/submit-code', methods=['POST'])
def submit_code():
    data = request.get_json()
    users_code = data['code']
    users_lang = data['lang']

    # TODO: do something here
    PISTON_EXECUTE_URL = 'https://emkc.org/api/v2/piston/execute'
    lang_to_file_name = {
        'py': 'main.py',
        'cpp': 'main.cpp',
        'java': 'Solution.java'
    }

    # request payload
    request_payload = {
        "language" : users_lang,
        "source" : users_code,
        "stdin" : "",
        "expected_output" : "",
    }

    results = []

    for test in test_cases:
        # Prepare input data for the test case
        input_data = json.dumps(test['input'])
        expected_output = json.dumps(test['expected'])

        # update stdin and expected output
        request_payload['stdin'] = input_data
        request_payload['expected_output'] = expected_output

        try:
            # send request to Piston API
            response = request.post(PISTON_EXECUTE_URL, json = request_payload)
            response_data = response.json()

            # check if it was successful
            if response_data.get('status') == 'success':
                user_output = response_data['output'].strip()
                user_output_json = json.loads(user_output)

                # compare the users output with the expected output
                if user_output_json == test['expected']:
                    results.append(f"The test case with input {test['input']} passed.")
                else:
                    results.append(f"The test case with input {test['input']} failed: expected {expected_output}, got {user_output}")
            else:
                results.append(f"Error executing code: {response_data.get('message', 'Unknown error')}")
        except Exception as e:
            results.append(f"Error running test case with input {test['input']}: {str(e)}")

    return jsonify({"message": "\n".join(results)})




def validate_code(users_code):
    return True # replcae with actual code validation logic using API

@app.route('/submit', methods=['POST'])
def submit_code():
    users_code = request.json.get('code')

    # Check if the user's code submission was correct
    is_correct = validate_code(users_code) # implement later
    if is_correct:
        return improve_code(users_code) # implement later
    else:
        return help_code(users_code) # implement later

@app.route('/improve', methods=['GET'])
def improve_code(users_code):
    # Pperform RAG for improvements
    prompt = f"Suggest improvements for the following code: {users_code}"
    suggestions = perform_rag(prompt)
    return jsonify({'message': suggestions})

@app.route('/help', methods=['GET'])
def help_code(users_code):
    # Perform RAG for help
    prompt = f"Suggest corrections to fix the following code: {users_code}"
    corrections = perform_rag(prompt)
    return jsonify({'message': corrections})

if __name__ == '__main__':
    app.run(debug=True)

    