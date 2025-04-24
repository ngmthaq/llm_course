from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
import os

# Load knowledge base
doc_loader = TextLoader(os.path.join(os.path.dirname(__file__), "test_document.txt"))
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
retriever = vector_store.as_retriever()

# Create PromptTemplate
prompt = ChatPromptTemplate.from_template(
    "\n".join(
        [
            "You are a helpful assistant.",
            "Please answer the question based on the context.",
            "CONTEXT: {context}",
            "QUESTION: {question}",
            "ANSWER: ",
        ]
    )
)

# Create LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "top_p": None,
        "top_k": None,
        "temperature": None,
    },
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("What is Higher Order Component?")
result = result.split("ANSWER:")[-1]
result = result.strip()
print(result)
