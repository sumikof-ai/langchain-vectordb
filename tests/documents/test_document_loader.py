import logging

from langchain_community.document_loaders import RecursiveUrlLoader

logger = logging.getLogger(__name__)

def test_recursive_loader():
    loader = RecursiveUrlLoader(
        "https://solana.com/ja/docs?locale=docs",
        max_depth=10
    )

    docs = loader.load()
    