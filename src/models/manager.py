import settings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class ModelsManager:
    def __init__(self, model_name:str=None):
        if not model_name:
            model_name = settings.EMBEDDINGS_MODEL
        model_path = f"{settings.MODEL_PATH}/{model_name}"
        print(model_path)
        self._embeddings = HuggingFaceEmbeddings(model_name=model_path,model_kwargs={
            "device": "cuda"
        })
    
    @property
    def embeddings(self):
        return self._embeddings