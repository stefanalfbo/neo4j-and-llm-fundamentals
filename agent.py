import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.tools import YouTubeSearchTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")


llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a movie expert. You find movies from a genre or plot.",
        ),
        ("human", "{input}"),
    ]
)

movie_chat = prompt | llm | StrOutputParser()


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


youtube = YouTubeSearchTool()


def call_trailer_search(input):
    input = input.replace(",", " ")
    return youtube.run(input)


# Create a custom tool - https://python.langchain.com/v0.2/docs/how_to/custom_tools/
tools = [
    Tool.from_function(
        name="Movie Chat",
        description="For when you need to chat about movies. The question will be a string. Return a string.",
        func=movie_chat.invoke,
    ),
    Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
    ),
]

# react - ReAct Reasoning and Acting
agent_prompt = hub.pull("hwchase17/react-chat")  # https://smith.langchain.com/hub/
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

while True:
    q = input("> ")

    response = chat_agent.invoke(
        {"input": q},
        {"configurable": {"session_id": SESSION_ID}},
    )

    print(response["output"])
