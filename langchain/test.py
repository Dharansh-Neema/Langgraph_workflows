from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

def testing(prompt):
    model = init_chat_model(model='google_genai:gemini-3.1-flash-lite')
    response = model.invoke(prompt)
    # print(response)
    return response.content[0]['text']

if __name__ == '__main__':
    print(testing("Hi what's capital of USA"))