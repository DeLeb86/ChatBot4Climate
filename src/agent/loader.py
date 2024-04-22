from langchain_community.document_loaders import JSONLoader
from pathlib import Path
import sys,os
sys.path.append(Path(__file__).parent.absolute().joinpath('../../').as_posix())
from settings import CRAWLING_OUTPUT_FOLDER, YEAR, SCHOOL
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_PATH = Path(__file__).parent.absolute().joinpath(
    f'../../{CRAWLING_OUTPUT_FOLDER}{SCHOOL}_courses_{YEAR}.json')

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["year"] = record.get("year")
    metadata["teachers"] = record.get("teachers")
    metadata["name"] = record.get("name")

    return metadata

if __name__ == '__main__':
    
    connection = "postgresql+psycopg://dba:dba@localhost:6024/chatbot"
    collection_name = "courses"

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema='.',
        content_key='total',
        metadata_func=metadata_func,
        json_lines=True
    )
        
    data = loader.load()
    #create store
    vectorstore = PGVector.from_documents(
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=connection,
        documents=data,
    )
    

    vectorstore.add_documents(data, ids=[doc.metadata["id"] for doc in data])
    
