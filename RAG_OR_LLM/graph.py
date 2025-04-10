from langgraph.graph import StateGraph,START,END
from retrival import generate
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from pydantic_structure import Router,AgentState
from typing import Literal
from IPython.display import Image, display
def router_agent(agentState):
    """This is router agent which will router between LLM or RAG"""
    try:
        llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-001",
                    api_key=os.getenv('GOOGLE_API_KEY'))
        llm_output = llm.with_structured_output(Router)
        query = agentState['query']
        
        prompt = """You have been given with this query:{query}, now analyse the query and decide whether to call
        RAG or LLM.
        It will return "RAG" if anything releated to Indian GDP is been asked It will return "LLM" if anything other than Indian GDP is beend asked so that the LLM can provide aproper and structured output to it."""
        formated_prompt = prompt.format(query=query)
        
        output = llm_output.invoke(formated_prompt)
        print(output)
        agentState['router'] = output.classification
        # return output.classification
        return agentState
    except Exception as e:
        print("Unexpected error occurred on router_agent",e)
        raise e

def llm_call(agentState):
    """This is LLM which will reply for any other query."""
    try:
        query = agentState['query']
        llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash-001",
                        api_key=os.getenv('GOOGLE_API_KEY'))
        output = llm.invoke(query)
        answer = agentState['answer']
        answer.append(output.content)
        agentState['answer'] = answer
        print(agentState)
        return agentState
    except Exception as e:
        print("Some unexpected error occurred at LLM CALL",e)
        raise e

def rag_call(agentState):
    """ This is RAG CALL """
    try:
        query = agentState['query']
        output = generate(question=query)
        answer = agentState['answer']
        answer.append(output)
        agentState['answer'] = answer
        print(agentState)
        return agentState
    
    except Exception as e:
        print("Some unexpected error occurred at RAG CALL",e)
        raise e
    
def invoke_graph(agentState):
    """This function will invoke the graph"""
    graph  = StateGraph(AgentState)
    graph = graph.add_node("router_agent",router_agent)
    graph = graph.add_node("LLM",llm_call)
    graph = graph.add_node("RAG",rag_call)
    graph = graph.set_entry_point("router_agent")
    graph = graph.add_conditional_edges("router_agent",
        lambda state : state['router'],
        {
            "RAG":"RAG",
            "LLM":"LLM"
        }
    )
    graph = graph.add_edge("RAG",END)
    graph = graph.add_edge("LLM",END)
    graph = graph.compile()
    # display(
    # Image(
    #     graph.get_graph().draw_mermaid_png()
    # ))
    # with open("graph_architecture.png", "wb") as f:
    #     f.write(graph.get_graph().draw_mermaid_png())
    graph.invoke(agentState)



if __name__ == "__main__":
    agentState = {}
    q = "What will be the GDP of India in 2025."
    q_l = "What is Langchain in 200 words?"
    agentState.update(query=q_l)
    agentState.update(answer=[])

    # print(AgentState)
    invoke_graph(agentState)