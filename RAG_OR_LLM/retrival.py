from langchain import hub
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
prompt = hub.pull("rlm/rag-prompt")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("indian-gdp")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = PineconeVectorStore(embedding=embeddings,index=index)
llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-001",
                api_key=os.getenv('GOOGLE_API_KEY'))
def retrieve(question):
   try:
      print("Inside retrieval")
      retrieved_docs = vector_store.similarity_search(query=question)
      return retrieved_docs
    #   print(retrieved_docs)
   except Exception as e:
      print("Some unexpected error occured while retriving",e)
      raise e
def generate(question:str):
    try:
        print("=====Inside Generate=======")
        context = retrieve(question=question)

        prompt_formatted = prompt.invoke({"question":question,"context":context})
     
        response = llm.invoke(prompt_formatted)
        return response.content
    except Exception as e:
       print("Unexpected error while geenrating the answer in RAG ",e)
       raise e
if __name__ =="__main__":
   gen = generate("Indian GDP in 2025?")
   print(gen)