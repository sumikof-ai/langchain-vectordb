import logging
import chromadb
import time
from langchain_chroma import Chroma
from models.manager import ModelsManager
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_text_splitters import CharacterTextSplitter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from langchain.globals import set_debug
set_debug(True)



persistent_client = chromadb.PersistentClient(path="chroma")

model_manager = ModelsManager()

vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="example_collection",
    embedding_function=model_manager.embeddings,
)
vector_store_from_client.reset_collection()


logger.info("loader.load() start")
loader = RecursiveUrlLoader(
        "https://solana.com/ja/docs", max_depth=10, extractor=lambda x: Soup(x, "html.parser").text
    )
docs = loader.load()
logger.info(f"len(docs) => {len(docs)}")


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
splitted_docs = text_splitter.split_documents(docs)
logger.info(f"len(splitted_docs) => {len(splitted_docs)}")


start_time = time.time()
logger.info("開始しました")
work_size = 1
for i in range(0,len(splitted_docs), work_size):
    logger.info(f"{i}->{i+work_size}まで取り込み開始 {time.time() - start_time:.6f}経過")
    vector_store_from_client.aadd_documents(splitted_docs[i:i+work_size])

result = vector_store_from_client.similarity_search_with_score(
    "solana accountについて教えてください。",k=10
)
result