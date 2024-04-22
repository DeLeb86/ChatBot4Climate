from langchain_community.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import HuggingFacePipeline
#from langchain_community.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer,AutoConfig, AutoModelForCausalLM,pipeline

model_name=''

model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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

retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
)


text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,                 
    return_full_text=True,
    max_new_tokens=300,
)

prompt_template = """
### [INST] 
Instruction: Answer the question based on the course database provided.
Here is context to help:

{context}

### QUESTION:
{question} 

[/INST]
 """

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template 
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain 
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
rag_chain = ( 
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
query=input()
print(rag_chain.invoke(query))

#docs = vectorstore.similarity_search(query, k=10)
#for d in docs:
#    print("id:",d.metadata["id"])
#    print("name:",d.metadata["name"])
#    print("teachers:",d.metadata["teachers"])
#    print(d.page_content)