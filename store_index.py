
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
import pinecone
from pinecone import ServerlessSpec
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()
index_name="agri-rag-bot"


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_pdf_file(data):
    loader_pdf=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents_pdf = loader_pdf.load()

    loader_csv=DirectoryLoader(data, glob="*.csv", loader_cls=CSVLoader)
    documents_csv = loader_csv.load()
    

    loader_txt=DirectoryLoader(data, glob="*.txt",loader_cls=TextLoader)
    documents_txt = loader_txt.load()

    documents = documents_csv+documents_pdf+documents_txt



    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        
    )
    texts = text_splitter.split_documents(extracted_data)
    return texts

text_chunks=text_split(extracted_data=extracted_data)

print(f"Total number of chunks: {len(text_chunks)}")

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')#'sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embeddings = download_hugging_face_embeddings()




pc=Pinecone(api_key=PINECONE_API_KEY)

from pinecone import ServerlessSpec
if not pc.has_index(name=index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    
index=pc.Index(name=index_name)
index.describe_index_stats()

documents = load_pdf_file('Data/')
text_chunks = text_split(documents)
embeddings = download_hugging_face_embeddings()
print("Proceeding with embeddings")

embedded_vectors = embeddings.embed_documents([chunk.page_content for chunk in text_chunks])
print("Embeddings generated")
def clean_metadata(meta: dict) -> dict:
    return {
        "text": meta.get("text", "")[:500],
        "source": meta.get("source", "unknown"),
        "page": meta.get("page", -1) if meta.get("page") is not None else -1
    }
# Prepare vectors for Pinecone
vectors = []
for i, (text_chunk, vector) in enumerate(zip(text_chunks, embedded_vectors)):
    
        metadata= {
            "text": text_chunk.page_content[:500],  # Keep metadata light!
            "source": text_chunk.metadata.get("source"),
            "page": text_chunk.metadata.get("page")
        }
        vectors.append({
        "id": f"chunk-{i}",
        "values": vector,
        "metadata": clean_metadata(metadata)
    })
    
print("Vectors prepared for Pinecone")
# === Batch Upload Safely ===
def batch_upsert(index, vectors, batch_size=50):
    for i in range(0, len(vectors), batch_size):
        print(f"Uploading batch {i // batch_size + 1} of {len(vectors) // batch_size + 1}")
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
# index_name="agri-rag-bot"
batch_upsert(index, vectors, batch_size=40)  # 40 is a safe number for 768 dim
print("✅ All chunks uploaded safely without exceeding Pinecone size limits.")

