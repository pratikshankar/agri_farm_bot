from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from pillow_heif import register_heif_opener
register_heif_opener()


def load_pdf_file(data):
    #loader=DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {data}")
    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        
    )
    print("RecursiveCharacterTextSplitter initialized")
    texts = text_splitter.split_documents(extracted_data)
    print(f"Split into {len(texts)} chunks")
    for doc in texts:
        doc.metadata["source"] = doc.metadata.get("source", "Unknown PDF")
        doc.metadata["page"] = doc.metadata.get("page", "Unknown Page")
    return texts

#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    # embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("Hugging Face Embeddings downloaded")
    return embeddings