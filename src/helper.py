from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#extract the data from pdf

def load_pdf_file(data):
    loader = DirectoryLoader(data,
                         glob="*.pdf", 
                         loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

#split the data into chunks

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    test_chunks = text_splitter.split_documents(extracted_data)
    return test_chunks

#download the embedding model for huggingface

def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings