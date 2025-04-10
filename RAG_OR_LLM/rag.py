from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging
import os 
#env variables
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(name="RAG")
# Load data
loader = TextLoader('./data/indian_gdp.txt')
data = loader.load()
logger.debug("Extracted the info")
print(data[0].page_content)


#Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#Vector db
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("Indian_GDP")
vector_store = PineconeVectorStore(embedding=embeddings,index=index)