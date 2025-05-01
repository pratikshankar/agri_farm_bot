from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.prompt import *
import os
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory


app = Flask(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")



embeddings = download_hugging_face_embeddings()


index_name = "agri-rag-bot"

# Embed each chunk and upsert the embeddings into your Pinecone index.

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
## initializing retriver object
retriver=docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 60})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    api_key=GEMINI_API_KEY,
    temperature=0.5,
    max_tokens=1024,
    timeout=None,
    max_retries=2
)



prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)



@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)