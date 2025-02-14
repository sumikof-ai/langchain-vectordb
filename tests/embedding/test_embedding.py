import chromadb
import logging
import tempfile

# import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture()
def vector_store():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        persistent_client = chromadb.PersistentClient(path=tempdir)
        embeddings = LlamafileEmbeddings(base_url="http://localhost:8080")

        vector_store_from_client = Chroma(
            client=persistent_client,
            collection_name="example_collection",
            embedding_function=embeddings,
        )
        yield vector_store_from_client


def test_vector_store(vector_store):
    example_texts = [
        "日本の桜の花見は春の風物詩です。",
        "富士山は日本の象徴であり、多くの観光客が訪れます。",
        "京都は伝統的な寺院や庭園が多く、日本文化を感じられる場所です。",
        "スマートフォンは現代社会で欠かせないツールです。",
        "AI技術は日々進化しており、さまざまな分野で活用されています。",
        "東京スカイツリーは、世界有数の高さを誇るタワーです。",
        "日本料理はヘルシーで、美味しいと世界中で人気があります。",
        "新幹線は速くて快適な移動手段として知られています。",
        "日本の温泉はリラックスするのに最適な場所です。",
        "お寿司は新鮮な魚を使った伝統的な日本料理の一つです。",
    ]
    documents = [Document(content, id=idx) for idx, content in enumerate(example_texts)]
    ids = [str(i) for i in range(len(example_texts))]  # 各データの一意のID

    # データをコレクションに追加
    vector_store.add_documents(documents=documents, ids=ids)

    results = vector_store.similarity_search_with_score(
        query="人工知能を使用した画期的なアイディアを教えてください。", k=3
        # query="日本文化を感じられる観光地はどこですか？", k=3
    )
    for result in results:
        logger.info(result)
