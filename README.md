# docBot
这是个能够阅读文档并回馈文档内容的AI，基于现有大模型以及Embedding模型等

## Usage
将文档置于`docs`目录下，运行`createIndex.py`：

```bash
python createIndex.py
```

然后根据所选模型运行相应程序，目前支持`Gemini`和`QWen`，已其一为例：

```bash
python QWen.py
```

等待模型加载后可发起对文档内容的提问

## Tips
- 大语言模型并不在本地运行，无需担心算力，但要保持联网
- 建议使用带有`CUDA`核心的`NVIDIA`显卡以加速Embedding模型运行
