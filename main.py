import chromadb
import pprint

# import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings import LlamafileEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

# embeddings = OllamaEmbeddings(base_url="http://localhost:11434")
embeddings = LlamafileEmbeddings(base_url="http://localhost:8080")

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name2")

vector_store_from_client  = Chroma(
    client=persistent_client,
    collection_name="example_collection",
    embedding_function=embeddings,
)

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
    "お寿司は新鮮な魚を使った伝統的な日本料理の一つです。"
]

# データをコレクションに追加
collection.add(
    documents=example_texts,  # テキストデータ
    # embeddings=embeddings.tolist(),  # ベクトルデータ
    ids=[str(i) for i in range(len(example_texts))]  # 各データの一意のID
)
print("aaaaaaa")
results = collection.query(
    query_texts=["日本文化を感じられる観光地はどこですか？"], # Chroma will embed this for you
    n_results=7 # how many results to return
)
print("aaaaaaa")
pprint.pp(results)
persistent_client.delete_collection("collection_name2")