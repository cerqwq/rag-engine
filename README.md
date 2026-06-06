# 🔎 RAG Engine

检索增强生成引擎，受ragflow (82k stars)启发，支持文档索引、语义检索、答案生成。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector-green" />
  <img src="https://img.shields.io/badge/OpenAI-API-purple" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

## ✨ 特性

- 📄 文档加载和分块
- 🔍 语义检索
- 🤖 LLM答案生成
- 📁 目录批量导入
- 💾 持久化存储

## 🚀 快速开始

```bash
pip install chromadb openai

python rag.py
```

## 📖 使用

```python
from rag import create_rag

# 创建RAG引擎
rag = create_rag(persist_dir="./my_rag")

# 添加文档
rag.add_document("Python是一种高级编程语言...", {"topic": "Python"})
rag.add_text_file("path/to/document.md")
rag.add_directory("path/to/docs/", extensions=[".md", ".txt"])

# 语义搜索
results = rag.search("Python编程")

# 生成答案
answer = rag.generate("什么是Python？")

# 对话式RAG
answer = rag.chat("Python有什么优势？")
```

## 📊 工作流程

```
文档 → 分块 → 向量化 → 存储
                         ↓
查询 → 向量检索 → 相关文档 → LLM生成 → 答案
```

## 📁 项目结构

```
rag-engine/
├── rag.py         # RAG引擎核心
├── vector_db/     # 向量数据库
└── README.md
```

## 📄 许可证

MIT License
