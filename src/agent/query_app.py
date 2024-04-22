from langchain_community.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd

import streamlit as st
import pandas as pd

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
def drop_tables():
    vectorstore.drop_tables()
    return {"message": "Tables dropped"}


def query(query: str,nb_results=5):
    print(query)
    docs = vectorstore.similarity_search_with_relevance_scores(query, k=nb_results*3)
    data=[(doc.metadata["id"],doc.metadata["name"],doc.metadata["teachers"],doc.page_content[:200],score) for doc,score in docs]
    df=pd.DataFrame(data,columns=["id","name","teachers","content","score"]).drop_duplicates(subset=["id"]).head(nb_results).reset_index(drop=True)
    return df 



st.set_page_config(page_title="Education4Climate Search Engine", layout="wide")
st.title("Education4Climate Search Engine")
query_input = st.text_input("Search information about ULB courses", value="")
nb_results=st.slider("Number of results", min_value=1, max_value=10, value=5,step=1)   
if query_input:
    df = query(query_input,nb_results=nb_results)
    st.write(df) 

    
    
    
    
    