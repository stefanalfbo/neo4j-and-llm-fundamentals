import os

from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser

load_dotenv()


chat_llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

memory = ChatMessageHistory()


def get_memory(session_id):
    return memory


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

current_weather = """
{
    "surf": [
        {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
        {"beach": "Polzeath", "conditions": "Flat and calm"},
        {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"},
        {"beach": "Perranporth", "conditions": "4ft waves and cross-shore winds"}
    ]
}"""

while True:
    question = input("> ")

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
            
        }, 
        config={
            "configurable": {"session_id": "none"}
        }
    )
    
    print(response)
    