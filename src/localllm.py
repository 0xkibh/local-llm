import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough

# Logging Config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


class LocalLLM:
    def __init__(self, model_name: str):
        self.documents = None
        self.PROMPT = None
        self.model_name = model_name
        self.embedding = OllamaEmbeddings(model=self.model_name)

    def read_document(self, document_path: str):
        loader = PyPDFLoader(document_path)
        documents = (
            loader.load_and_split()
        )  # Default chunk_size: int = 4000, chunk_overlap: int = 200,
        self.documents = documents
        logging.info("Document Fetched and Converted to Chunks")

    def document_to_embeddings(self):
        """Convert to Vector and Save Vector"""
        db = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embedding,
        )
        db.save_local("faiss_index")
        logging.info("Document Chunks converted to Embeddings and Saved")

    def create_prompt(self):
        prompt_template = """Use the following pieces of context to answer the question at the end.

        {context}

        Question: {question}
        Answer:"""
        self.PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def nearest_neighbors(self, query: str, k: int = 3):
        db = FAISS.load_local(
            "faiss_index", self.embedding, allow_dangerous_deserialization=True
        )
        docs = db.similarity_search(query, k=k)
        # create chain with model and prompt template
        return docs

    def run(self, query: str) -> None:
        self.create_prompt()
        docs = self.nearest_neighbors(query=query)
        llm = Ollama(model=self.model_name)
        chain = (
            {"context": lambda x: docs, "question": RunnablePassthrough()}
            | self.PROMPT
            | llm
        )
        response = chain.invoke(query)
        print(response)
