import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA

# from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
embedding_provider = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

k = 4  # Number of similar documents to retrieve. The default is 4.
result = movie_plot_vector.similarity_search(
    "A movie where aliens land and attack earth.", k=k
)

print("Retrieved documents:\n")

for doc in result:
    print(doc.metadata["title"], "-", doc.page_content)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True,
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print("\nRetrievalQA:\n")
print(response)

# Create a new vector index - https://graphacademy.neo4j.com/courses/llm-vectors-unstructured/

# # A list of Documents
# documents = [
#     Document(
#         page_content="Text to be indexed",
#         metadata={"source": "local"}
#     )
# ]

# new_vector = Neo4jVector.from_documents(
#     documents,
#     embedding_provider,
#     graph=graph,
#     index_name="myVectorIndex",
#     node_label="Chunk",
#     text_node_property="text",
#     embedding_node_property="embedding",
#     create_id_index=True,
# )
