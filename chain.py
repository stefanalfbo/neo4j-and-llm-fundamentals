import os

from dotenv import load_dotenv
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()


llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


template = PromptTemplate.from_template("""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Output JSON as {{"description": "your response here"}}

Tell me about the following fruit: {fruit}
""")

llm_chain = template | llm | SimpleJsonOutputParser()

response = llm_chain.invoke({"fruit": "apple"})

print(response)
