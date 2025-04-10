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
    try:
        query = AgentState['query']
        llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-001",
                        api_key=os.getenv('GOOGLE_API_KEY'))
        output = llm.invoke(query)
        print(output.content)
        answer = AgentState['answer']
        answer.append(output.content)
        AgentState['answer'] = answer
        print(AgentState)
        return AgentState
    except Exception as e:
        print("Some unexpected error occurred at LLM CALL",e)
        raise e

def rag_call(AgentState):
    """ This is RAG CALL """
    try:
        query = AgentState['query']
        output = generate(question=query)
        answer = AgentState['answer']
        answer.append(output)
        AgentState['answer'] = answer
        print(AgentState)
        return AgentState
    
    except Exception as e:
        print("Some unexpected error occurred at RAG CALL",e)
        raise e

if __name__ == "__main__":
    AgentState = {}
    q = "What will be the GDP of India in 2025."
    q_l = "What is Langchain?"
    AgentState.update(query=q)
    AgentState.update(answer=[])
    # print(AgentState)
    rag_call(AgentState)