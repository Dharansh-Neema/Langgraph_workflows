from pydantic import BaseModel,Field
from typing import Literal,TypedDict,List
class Router(BaseModel):
    reasoning : str = Field(description="This contain the reasoning behind the decision taken")
    classification : Literal["LLM","RAG"]  = Field(description="""This field will return either LLM or RAG based on the provided query
                                                It will return "RAG" if anything releated to Indian GDP is been asked
                                                It will return "LLM" if anything other than Indian GDP is beend asked so that the LLM can provide a
                                                proper and structured output to it.""")
    
class AgentState(TypedDict):
    query:str
    answer:List
    router:str