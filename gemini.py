import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from models import RerankingModel, EmbeddingModel, BM25Model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

api_key = "YOUR_API_KEY"
base_url = None
llm_model = "gemini-pro"

with open('index/texts.json', 'r', encoding='utf-8') as f:
    texts = json.load(f)

embed1 = EmbeddingModel('thenlper/gte-large-zh', 'index/embed1.index')
embed2 = EmbeddingModel('BAAI/bge-large-zh-v1.5', 'index/embed2.index')
bm25 = BM25Model(texts) 
models = [embed1, embed2, bm25]

ranker_path = 'BAAI/bge-reranker-large'
ranker = RerankingModel(ranker_path)

llm = ChatGoogleGenerativeAI(model=llm_model, api_key=api_key)

while True:
    query = input("问题：")
    context = []
    for model in models:
        indices = model.query(query)
        context += [texts[index] for index in indices]

    content = "上下文：" + "\n\n".join(ranker.select(query, context)) + "\n\n问题：" + query

    messages = [
        SystemMessage(content="请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。如果所给材料与用户问题无关，只输出：无答案。"),
        HumanMessage(content=content),
    ]

    parser = StrOutputParser()
    result = llm.invoke(messages)
    print("回答：" + parser.invoke(result) + '\n')