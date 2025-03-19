import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

llm = GoogleGenerativeAI(
    #model="gemini-2.0-flash-lite",
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embedding_provider = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    model="models/text-embedding-004"
    )

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

chunk_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="chunkVector",
    embedding_node_property="textEmbedding",
    text_node_property="text",
    retrieval_query="""
        // get the document
        MATCH (node)-[:PART_OF]->(d:Document)
        WITH node, score, d

        // get the entities and relationships for the document
        MATCH (node)-[:HAS_ENTITY]->(e)
        MATCH p = (e)-[r]-(e2)
        WHERE (node)-[:HAS_ENTITY]->(e2)

        // unwind the path, create a string of the entities and relationships
        UNWIND relationships(p) as rels
        WITH
            node,
            score,
            d,
            collect(apoc.text.join(
                [labels(startNode(rels))[0], startNode(rels).id, type(rels), labels(endNode(rels))[0], endNode(rels).id]
                ," ")) as kg
        RETURN
            node.text as text, score,
            {
                document: d.id,
                entities: kg
            } AS metadata
        """
)

instructions = (
    "Use the given context to answer the question."
    "Reply with an answer that includes the id of the document and other relevant information from the text."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

chunk_chain = create_stuff_documents_chain(llm, prompt)
chunk_retriever = chunk_vector.as_retriever()
chunk_retrieval_chain = create_retrieval_chain(
    chunk_retriever,
    chunk_chain
)

def find_chunk(q):
    return chunk_retrieval_chain.invoke({"input": q})


while (q := input("> ")) != "exit":
    print(find_chunk(q))
