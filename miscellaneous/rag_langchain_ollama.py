# Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# Langchain Community
from langchain_community.document_loaders import TextLoader

# Langchain Core
from langchain_core.vectorstores import InMemoryVectorStore

# Langchain Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Other
from os import name, system
from os.path import join, dirname
from json import dumps


class App:
    def __init__(self):
        self.history = []
        self.model_name = "gemma3:4b"
        self.embedding_name = "all-minilm:latest"
        self.temperature = 0.5
        self.txt_file = join(dirname(__file__), "./rag_langchain_ollama.txt")

    def clear_stdout(self):
        system('cls' if name == 'nt' else 'clear')

    def choose_temperature(self):
        answer = input("\n".join([
            "🤖: What is your expectation about creative level from AI Model?",
            "       1. Strict",
            "       2. Normal (Default)",
            "       3. Creative",
            "😑: ",
        ]))
        if answer == "1":
            self.temperature = 0.1
        elif answer == "3":
            self.temperature = 0.9
        else:
            self.temperature = 0.5

    def init_llm(self):
        print(
            f"🤖: Initializing LLM from Ollama with '{self.model_name}' and temperature is {self.temperature}")
        self.llm = OllamaLLM(
            model=self.model_name,
            temperature=self.temperature
        )

    def init_multi_queries_llm(self):
        print(
            f"🤖: Initializing Multi-Queries LLM from Ollama with '{self.model_name}'")
        self.mq_llm = OllamaLLM(
            model=self.model_name,
            temperature=0.9,
        )

    def init_embedding(self):
        print(
            f"🤖: Initializing Embedding from Ollama with '{self.embedding_name}'")
        self.embedding = OllamaEmbeddings(model=self.embedding_name)

    def load_doc(self):
        print("🤖: Loading document")
        text_loader = TextLoader(self.txt_file)
        doc = text_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        print("🤖: Splitting document into chunks")
        self.chunks = text_splitter.split_documents(doc)

    def init_vector_store(self):
        print("🤖: Initializing vector store")
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=self.chunks,
            embedding=self.embedding,
        )

    def init_prompt_template(self):
        self.prompt_template = ChatPromptTemplate.from_template(
            "\n\n".join(
                [
                    "You are a helpful assistant.",
                    "Please answer the question based on the following context and chat history.",
                    "CONTEXT: {context}.",
                    "HISTORY: {history}",
                    "QUESTION: {question}.",
                ]
            )
        )

    def convert_to_multi_query(self, question):
        prompt_template = ChatPromptTemplate.from_template(
            "\n\n".join([
                "You are a helpful assistant.",
                "Your task is generate exactly 5 different questions from original question.",
                "Each question should be in a separate line.",
                "ORIGINAL QUESTION: {question}.",
            ])
        )
        prompt = prompt_template.invoke({"question": question})
        response = self.mq_llm.invoke(prompt.to_string())
        response = response.split("\n")
        return response

    def search_knowledge(self, question):
        retriever = self.vector_store.as_retriever()
        multi_queries = self.convert_to_multi_query(question)
        multi_queries.append(question)
        context = []
        for query in multi_queries:
            query = query.strip()
            if query:
                docs = retriever.invoke(query)
                content = "\n".join([doc.page_content for doc in docs])
                context.append(content)
        return "\n".join(set(context))

    def ask(self):
        question = input("🤖: What is your question?\n😑: ")
        if question == "exit":
            print("🤖: Bye!")
            return

        context = self.search_knowledge(question)
        history = dumps(self.history)
        params = {"context": context, "history": history, "question": question}
        prompt = self.prompt_template.invoke(params)
        response = self.llm.invoke(prompt.to_string())
        self.history.append(f"Human: {question}")
        self.history.append(f"Ai: {response}")
        print(f"🤖: {response}")
        self.ask()

    def invoke(self):
        self.clear_stdout()
        self.choose_temperature()
        self.init_multi_queries_llm()
        self.init_llm()
        self.init_embedding()
        self.load_doc()
        self.init_vector_store()
        self.init_prompt_template()
        self.ask()


App().invoke()
