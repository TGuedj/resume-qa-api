# FILE: app/qa_chain.py

import os
from langchain_ollama import OllamaLLM as Ollama
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Get the Ollama host from an environment variable, defaulting to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:11434"

# Initialize the components that don't depend on the request
llm = Ollama(model="llama3", base_url=OLLAMA_BASE_URL)
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step-by-step. If you don't know the answer, just say that you don't know.

<context>
{context}
</context>

Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)


def get_qa_chain_for_candidate(vectorstore: Chroma, candidate_id: str):
    """
    Creates a RAG chain filtered for a specific candidate.

    Args:
        vectorstore: The ChromaDB vector store instance.
        candidate_id (str): The ID of the candidate to filter for.

    Returns:
        The filtered RAG chain.
    """
    # Create a retriever that filters by candidate_id in the metadata
    retriever = vectorstore.as_retriever(
        search_kwargs={'filter': {'candidate_id': candidate_id}}
    )

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
