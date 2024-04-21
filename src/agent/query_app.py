from langchain_community.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from fastapi import FastAPI 

#setup resources
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

#create store
vectorstore = PGVector(
    embedding_function=embeddings,
    collection_name=collection_name,
    connection_string=connection,
    distance_strategy=DistanceStrategy.COSINE,
    
)
app = FastAPI() 

@app.get("/drop_tables")
def drop_tables():
    vectorstore.drop_tables()
    return {"message": "Tables dropped"}


@app.post("/query")
def query(query: str,nb_results=5):
    docs = vectorstore.similarity_search(query, k=nb_results)
    print(docs)
    return docs 

    
    
    
    
    