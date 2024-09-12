import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

template = PromptTemplate(
    template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""",
    input_variables=["fruit"],
)

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0,
)

response = llm.invoke("What is Neo4j?")

print(response)

response = llm.invoke(template.format(fruit="apple"))

print(response)
