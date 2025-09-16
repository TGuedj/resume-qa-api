from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_document(file_path: str):
    """
    Loads a PDF document from the given file path and splits it into chunks.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of document chunks.
    """
    print(f"Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print(f"Document split into {len(splits)} chunks.")
    return splits