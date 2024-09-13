import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    Neo4jChatMessageHistory,
)
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


SESSION_ID = str(uuid4())

print(f"""Session Id: {SESSION_ID}""")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Test connection with a cypher query
# result = graph.query("""
# MATCH (m:Movie{title: 'Toy Story'})
# RETURN m.title, m.plot, m.poster
# """)

# print(result)

# Show schema information
# print(graph.schema)

# Refreshing the schema
# graph.refresh_schema()

chat_llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

memory = ChatMessageHistory()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


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
            {"beach": "Bells", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

while True:
    question = input("> ")

    response = chat_with_message_history.invoke(
        {
            "context": current_weather,
            "question": question,
        },
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print(response)
