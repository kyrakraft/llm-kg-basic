import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

load_dotenv()

DOCS_PATH = "./data/pdfs"

llm = GoogleGenerativeAI(
    #model="gemini-2.0-flash-lite",
    model="models/gemini-1.5-pro-001",
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

doc_transformer = LLMGraphTransformer(
    llm=llm,
    #allowed_nodes=["Concept", "Physical Entity"],
    #node_properties=["name", "description"],
    )

loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    #add document and chunk nodes to kg
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """,
        properties
    )

    #generate entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])
    print("hi1")
    #map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        print("hi2")
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node,
                    type="HAS_ENTITY"
                    )
                )

    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)
    print(graph)

#create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
    }};""")

print("Neo4j Response:", result)
