from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models import EmbeddingModel
import torch
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
embed1 = EmbeddingModel('thenlper/gte-large-zh')
embed2 = EmbeddingModel('BAAI/bge-large-zh-v1.5')

if __name__ == '__main__':
    print(torch.cuda.is_available())
    loader = DirectoryLoader('docs', show_progress=True, use_multithreading=True)
    document = loader.load()

    chunk_size = 512
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators = ["\n\n", "\u3000\u3000", "\n", "。", " "])
    chunks = text_splitter.split_documents(document)
    texts = [chunk.page_content for chunk in chunks]

    if not os.path.exists('index'):
        os.mkdir('index')
    embed1.save_index(texts, 'index/embed1.index')
    embed2.save_index(texts, 'index/embed2.index')

    with open('index/texts.json', 'w', encoding='utf-8') as f:
        json.dump(texts, f)