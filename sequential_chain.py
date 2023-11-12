import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()  # take environment variables from .env.


llm = OpenAI(
    openai_api_key=os.getenv('OPENAI_KEY')
)


code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)


test_prompt = PromptTemplate(
    template="Write a test for the following {language} code \n{code}",
    input_variables=["language", "code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"  # Gives output with a key called code instead of usual "text" key
)


test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)


chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]
)

result = chain(
    {
        "language": "python",
        "task": "return a list of numbers"
    }
)

print(f"Language: {result['language']}\n")
print(f"Task: {result['task']}\n")
print(f"Test: {result['test']}\n")
print(f"Code: {result['code']}\n")

