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


# Create prompt for multi query
mq_prompt = ChatPromptTemplate.from_template(
    " ".join(
        [
            "You are an AI language model assistant.",
            "Your task is to generate exactly 3 different, each on a separate line, but related search queries that will help retrieve relevant documents from a vector database.",
            "USER QUESTION: {question}.",
            "DIFFERENT QUESTIONS:",
        ]
    )
)

# Create multi query LLM
mq_llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 128,
        "do_sample": False,
        "top_p": None,
        "top_k": None,
        "temperature": 0,
    },
)

# Create multi query chain
mq_chain = {"question": RunnablePassthrough()} | mq_prompt | mq_llm | StrOutputParser()

questions = mq_chain.invoke("What is Higher Order Component?")
questions = questions.split("DIFFERENT QUESTIONS:")[-1]
questions = questions.strip()
questions = questions.split("\n")

# # Create prompt for question answering
# qa_prompt = ChatPromptTemplate.from_template(
#     "\n".join(
#         [
#             "You are a helpful assistant.",
#             "Please answer the question based on the context.",
#             "CONTEXT: {context}",
#             "QUESTION: {question}",
#             "ANSWER:",
#         ]
#     )
# )

# # Create QA LLM
# qa_llm = HuggingFacePipeline.from_model_id(
#     model_id="google/gemma-3-1b-it",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 256,
#         "do_sample": False,
#         "top_p": None,
#         "top_k": None,
#         "temperature": None,
#     },
# )

# # Create QA chain
# qa_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | qa_prompt
#     | qa_llm
#     | StrOutputParser()
# )

# result = qa_chain.invoke("What is hoisting?")

print(questions)
