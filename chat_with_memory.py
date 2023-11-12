import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()  # take environment variables from .env.

chat_model = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_KEY')
)


chat_prompt = ChatPromptTemplate(
    input_variables=["user_input"],
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful but sassy AI assistant, who helps people."),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ]
)

chat_chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt,
    output_key="chat_response"  # Gives output with a key called code instead of usual "text" key
)


user_input = ''
while user_input != 'q':
    user_input = input(">> ")
    result = chat_chain({
            "user_input": user_input,
    })
    print(result)
