# In the route handlers, perform the RAG process to retrieve
# suggestions from MongoDB based on the user's code submission

from flask import Flask, request, jsonify
from pymongo import MongoClient
import json

import subprocess

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nest_asyncio


from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

nest_asyncio.apply()

openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# MongoDB setup
client = MongoClient(mongo_connection_string)
db = client['leetcode']
collection = db['solutions']

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# Load documents from MongoDB and prepare for RAG
def load_and_prepare_docs():
    loader = MongodbLoader(
        connection_string=mongo_connection_string,
        db_name="leetcode",
        collection_name="solutions",
        field_names=["solution", "problem_id"]
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define the RAG chain (same as the notebook)
def build_rag_chain(retriever):
    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Example test cases
test_cases = [
    {
        "input": [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]],
        "expected": [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
    }
]

# When the user submits code check against the test cases
# @app.route('/submit-code', methods=['POST'])
# def submit_code():
#     data = request.get_json()
#     users_code = data['code']
#     users_lang = data['lang']

#     # TODO: do something here
#     PISTON_EXECUTE_URL = 'https://emkc.org/api/v2/piston/execute'
#     lang_to_file_name = {
#         'py': 'main.py',
#         'cpp': 'main.cpp',
#         'java': 'Solution.java'
#     }

#     # request payload
#     request_payload = {
#         "language" : users_lang,
#         "source" : users_code,
#         "stdin" : "",
#         "expected_output" : "",
#     }

#     results = []

#     for test in test_cases:
#         # Prepare input data for the test case
#         input_data = json.dumps(test['input'])
#         expected_output = json.dumps(test['expected'])

#         # update stdin and expected output
#         request_payload['stdin'] = input_data
#         request_payload['expected_output'] = expected_output

#         try:
#             # send request to Piston API
#             response = request.post(PISTON_EXECUTE_URL, json = request_payload)
#             response_data = response.json()

#             # check if it was successful
#             if response_data.get('status') == 'success':
#                 user_output = response_data['output'].strip()
#                 user_output_json = json.loads(user_output)

#                 # compare the users output with the expected output
#                 if user_output_json == test['expected']:
#                     results.append(f"The test case with input {test['input']} passed.")
#                 else:
#                     results.append(f"The test case with input {test['input']} failed: expected {expected_output}, got {user_output}")
#             else:
#                 results.append(f"Error executing code: {response_data.get('message', 'Unknown error')}")
#         except Exception as e:
#             results.append(f"Error running test case with input {test['input']}: {str(e)}")

#     return jsonify({"message": "\n".join(results)})

retriever = load_and_prepare_docs()
rag_chain = build_rag_chain(retriever)

def validate_code(users_code):
    return True # replcae with actual code validation logic using API

@app.route('/submit', methods=['POST'])
def submit_code():
    users_code = request.json.get('code')

    # Check if the user's code submission was correct
    is_correct = validate_code(users_code)
    if is_correct:
        return improve_code(users_code) 
    else:
        return help_code(users_code) 

@app.route('/improve', methods=['POST'])
def improve_code():
    print('POST /improve request')
    # Pperform RAG for improvements
    users_code = request.json.get('users_code')
    prompt = f"Suggest improvements for the following code: {users_code}"
    suggestions = rag_chain.invoke(prompt)
    return jsonify({'message': suggestions})

@app.route('/help', methods=['POST'])
def help_code(users_code):
    # Perform RAG for help
    prompt = f"Suggest corrections to fix the following code: {users_code}"
    corrections = rag_chain.invoke(prompt)
    return jsonify({'message': corrections})

if __name__ == '__main__':
    app.run(debug=True)

    