### Import Section ###
import json
import uuid
import os
from operator import itemgetter

from redis import Redis
from dotenv import load_dotenv
import chainlit as cl

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_community.cache import RedisSemanticCache
from langchain.globals import set_llm_cache
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.schema.runnable.config import RunnableConfig

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


### Global Section ###

# Models
print("Initializing models..")
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
chat_model = ChatOpenAI(model="gpt-4o-mini")
core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Prompts
print("Setting up prompts...")
rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""
rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]
rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])

# Docs - these are chunked in advance
print("Reading chunked docs...")
def read_docs_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return [Document(**doc) for doc in data]
docs = read_docs_from_file("chunked_docs.json")

# Semantic cache with Redis - must be running in the container!
print("Building semantic cache...")
redis_host = "redis"
redis_port = 6379
redis_client = Redis(host=redis_host, port=redis_port)

set_llm_cache(RedisSemanticCache(
    redis_url=f"redis://{redis_host}:{redis_port}",
    embedding=core_embeddings
))

# Vector store with Qdrant
print("Initializing vector store...")
collection_name = f"pdf_to_parse_{uuid.uuid4()}"
#client = QdrantClient("qdrant")
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
new_vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=core_embeddings)

new_vectorstore.add_documents(docs)
new_retriever = new_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    print("Entering on_chat_start...")
    # Set chain
    rag_chain = (
        {"context": itemgetter("question") | new_retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | chat_model
    )
    cl.user_session.set("lcel_rag_chain", rag_chain)

    # Give a sign of life
    await cl.Message(content="I'm ready to chat now!").send()


### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(lcel_rag_chain.stream)(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        #print(f"chunk: {chunk}, type(chunk): {type(chunk)}")
        await msg.stream_token(chunk.content)

    await msg.send()