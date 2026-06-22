from langchain.chat_models import init_chat_model
from langchain.tools import tool
from datetime import datetime
from dotenv import load_dotenv
from typing import List
load_dotenv()

model = init_chat_model(model='google_genai:gemini-3.1-flash-lite')
def testing(prompt):
   
    response = model.invoke(prompt)
    # print(response)
    return response.content[0]['text']

def streaming(prompt):
    
    response = model.stream(prompt)
    for chunk in response:
        print(chunk.text,end="",flush=True)


def batching(prompts: List):
    print(model.batch(prompts))


# To create a tool 
@tool
def date_time_return():
    """
    Return Exact date and time
    """
    print(datetime.now())
    return datetime.now()
if __name__ == '__main__':
    # print(testing("Hi what's capital of USA"))
    # streaming("Describe about banswara a city in rajasthan in 1000 words ")
    prompts = [
        "What should a Associate software enginner do to get hired as AI enginner in TOP fintech or banks",
        "Give some good projects ideas for AI-Enginner Industry standard types"
    ]
    # batching(prompts)
    # To Bind the models with tools 
    model_with_date_tool = model.bind_tools([date_time_return])
    print(model_with_date_tool.invoke('What todays date??'))