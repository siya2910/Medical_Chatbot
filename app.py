from flask import Flask,render_template,jsonify,request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from lanchain.chains import create_retrieval_chain
from lanchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os 

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "medical-bot"

#embed each chunk

docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
)

retrieved = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retrieved, question_answer_chain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['GET','POST'])
def chat():
    msg = request.form("msg")
    input = "msg"
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response:",response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port =8080,debug=True)