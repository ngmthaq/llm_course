from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load knowledge base
doc_loader = TextLoader(os.path.join(os.path.dirname(__file__), "./test_document.txt"))
doc = doc_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split the document into chunks
chunks = text_splitter.split_documents(doc)

# Create a vector store from the chunks
# Chunks are embedded using the HuggingFaceEmbeddings model
# and stored in an in-memory vector store
vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
)

# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 1})
