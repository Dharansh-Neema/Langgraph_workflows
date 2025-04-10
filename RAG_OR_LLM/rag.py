from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# print(data[0].page_content)
cont = data[0].page_content
#Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#Vector db
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# pc.create_index(name="indian-gdp",
#                 dimension=768,
#                 metric="cosine",
#                 spec={
#                     "serverless": {
#                             "region": "us-east-1",
#                             "cloud": "aws"}})
index = pc.Index("indian-gdp")
vector_store = PineconeVectorStore(embedding=embeddings,index=index)

#Text-Splitter 
text_spliter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
splits = text_spliter.split_documents(data)
# print(splits)
logger.debug("Implemented text splitter successfully")

# Vector store 
doc_ids = vector_store.add_documents(splits)
# print(doc_ids[:5])
logger.debug("Successfully stored the result")