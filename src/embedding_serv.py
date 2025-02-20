from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from typing import List
import threading

import uvicorn

import settings


class EmbeddingModel:
    """
    シングルトンで管理される埋め込みモデルクラス
    排他制御を用いてスレッドセーフな埋め込み取得を実現
    """

    _instance = None
    _lock = threading.Lock()  # モデルのロック

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:  # 初期化時の排他制御
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    model_path = f"{settings.MODEL_PATH}/{settings.EMBEDDINGS_MODEL}"
                    cls._instance.model = HuggingFaceEmbeddings(
                        model_name=model_path, model_kwargs={"device": "cuda"}
                    )
        return cls._instance

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        with self._lock:  # 埋め込み生成時の排他制御
            return self.model.embed_documents(texts)


# FastAPI アプリケーションの初期化
app = FastAPI()


@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc: RequestValidationError):
    print(exc)
    print(await request.body())
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


# OpenAI互換のリクエストボディ
class EmbeddingRequestBaseOpenAI(BaseModel):
    model: str
    input: List[str]


# OpenAI互換のレスポンスボディ
class EmbeddingResponseBaseOpenAI(BaseModel):
    object: str
    data: List[dict]
    model: str
    usage: dict


@app.post("/v1/embeddings", response_model=EmbeddingResponseBaseOpenAI)
def v1_embeddings(request):
    # def get_embeddings(request: EmbeddingRequest):
    # シングルトンの埋め込みモデルインスタンスを取得
    embedding_model = EmbeddingModel()
    print(request)
    # テキストを埋め込みに変換（排他制御付き）
    embeddings = embedding_model.get_embedding(request.input)

    # OpenAI互換のレスポンスフォーマット
    response = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": embedding}
            for i, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in request.input),
            "total_tokens": sum(len(text.split()) for text in request.input),
        },
    }
    print(response)
    return response


# Llamafile互換のリクエストボディ
class EmbeddingRequestBaseLlama(BaseModel):
    content: str


# Llamafile互換のレスポンスボディ
class EmbeddingResponseBaseLlama(BaseModel):
    embedding: List[float]


@app.post("/embedding", response_model=EmbeddingResponseBaseLlama)
def embedding(request: EmbeddingRequestBaseLlama):
    # シングルトンの埋め込みモデルインスタンスを取得
    embedding_model = EmbeddingModel()
    # テキストを埋め込みに変換（排他制御付き）
    embeddings = embedding_model.get_embedding([request.content])

    # Llamafile互換のレスポンスフォーマット
    response = {"embedding": embeddings[0]}
    return response


if __name__ == "__main__":
    uvicorn.run(app=app)
