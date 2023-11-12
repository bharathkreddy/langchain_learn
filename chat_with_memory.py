import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
load_dotenv()  # take environment variables from .env.

chat_model = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_KEY')
)


memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),  # saves all the memory to a file, langchain allows dbs too.
    memory_key="messages",
    return_messages=True  # for conv.Buffer Memory to return actual messages and not strings
)


chat_prompt = ChatPromptTemplate(
    input_variables=["user_input", "messages"],  # Add "messages" to input variables.
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful but sassy AI assistant, who helps people."),
        MessagesPlaceholder(variable_name="messages"),  # Placeholder for all memory messages.
        HumanMessagePromptTemplate.from_template("{user_input}")
    ]
)

chat_chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt,
    memory=memory,
    output_key="chat_response"  # Gives output with a key called code instead of usual "text" key
)


user_input = ''
while user_input != 'q':
    user_input = input(">> ")
    result = chat_chain({
            "user_input": user_input,
    })
    print(result['chat_response'])
