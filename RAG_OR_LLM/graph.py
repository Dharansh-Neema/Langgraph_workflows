from langgraph.graph import StateGraph
from retrival import generate
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from pydantic_structure import Router
from typing import Literal

def router_agent(AgentState)->Literal["LLM","RAG"]:
    """This is router agent which will router between LLM or RAG"""
    try:
        llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    api_key=os.getenv('GOOGLE_API_KEY'))
        llm_output = llm.with_structured_output(Router)
        query = AgentState['query']
        
        prompt = """You have been given with this query:{query}, now analyse the query and decide whether to call
        RAG or LLM.
        It will return "RAG" if anything releated to Indian GDP is been asked It will return "LLM" if anything other than Indian GDP is beend asked so that the LLM can provide aproper and structured output to it."""
        formated_prompt = prompt.format(query=query)
        
        output = llm_output.invoke(formated_prompt)
        print(output)
        return output
    except Exception as e:
        print("Unexpected error occurred on router_agent",e)
        raise e

def llm_call(AgentState):
    """This is LLM which will reply for any other query."""

if __name__ == "__main__":
    AgentState = {}
    q = "What will be the GDP of India in 2025."
    AgentState.update(query=q)
    print(AgentState)
    router_agent(AgentState)