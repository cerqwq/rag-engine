"""
RAG Engine - 检索增强生成引擎
受 ragflow (82k stars) 启发，支持文档索引、语义检索、答案生成
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Document:
    """文档"""
    def __init__(self, content: str, metadata: Dict = None, doc_id: str = None):
        self.id = doc_id or f"doc_{int(datetime.now().timestamp() * 1000)}"
        self.content = content
        self.metadata = metadata or {}
        self.chunks: List[str] = []

    def chunk(self, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """分块"""
        words = self.content.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        self.chunks = chunks
        return chunks


class VectorStore:
    """向量存储"""
    def __init__(self, collection_name: str = "documents", persist_dir: str = None):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb未安装")

        path = persist_dir or "./vector_db"
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, doc_id: str, content: str, metadata: Dict = None):
        """添加文档"""
        self.collection.add(
            documents=[content],
            ids=[doc_id],
            metadatas=[metadata or {}]
        )

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """语义搜索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        return [
            {"id": id, "content": doc, "score": score, "metadata": meta}
            for id, doc, score, meta in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            )
        ]

    def delete(self, doc_id: str):
        """删除文档"""
        try:
            self.collection.delete(ids=[doc_id])
        except:
            pass

    def clear(self):
        """清空"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """文档数量"""
        return self.collection.count()


class RAGEngine:
    """
    RAG引擎
    支持：文档加载、分块、索引、检索、生成
    """

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str = None,
        base_url: str = None,
        persist_dir: str = None
    ):
        self.model = model
        self.persist_dir = persist_dir or "./rag_data"

        # 向量存储
        self.store = VectorStore("rag_documents", self.persist_dir)

        # LLM客户端
        if OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=api_key or os.environ.get('OPENAI_API_KEY', ''),
                base_url=base_url or os.environ.get('OPENAI_BASE_URL', 'https://api.xiaomimimo.com/v1')
            )
        else:
            self.client = None

        self.documents: Dict[str, Document] = {}

    def add_document(self, content: str, metadata: Dict = None) -> str:
        """添加文档"""
        doc = Document(content, metadata)
        doc.chunk()

        # 添加到向量存储
        for i, chunk in enumerate(doc.chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            chunk_metadata = {
                "doc_id": doc.id,
                "chunk_index": i,
                **(metadata or {})
            }
            self.store.add(chunk_id, chunk, chunk_metadata)

        self.documents[doc.id] = doc
        return doc.id

    def add_text_file(self, file_path: str, metadata: Dict = None) -> str:
        """添加文本文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        meta = {"source": str(path), "filename": path.name, **(metadata or {})}
        return self.add_document(content, meta)

    def add_directory(self, dir_path: str, extensions: List[str] = None) -> List[str]:
        """添加目录中的所有文本文件"""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        extensions = extensions or [".txt", ".md", ".py", ".js", ".json"]
        doc_ids = []

        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    doc_id = self.add_text_file(str(file_path))
                    doc_ids.append(doc_id)
                except Exception as e:
                    print(f"Failed to add {file_path}: {e}")

        return doc_ids

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """语义搜索"""
        return self.store.search(query, limit)

    def generate(self, query: str, context_limit: int = 3) -> str:
        """检索并生成答案"""
        if not self.client:
            return "LLM客户端未配置"

        # 检索相关文档
        relevant = self.search(query, limit=context_limit)

        if not relevant:
            return "未找到相关文档"

        # 构建上下文
        context = "\n\n".join([
            f"[来源: {r['metadata'].get('source', '未知')}]\n{r['content']}"
            for r in relevant
        ])

        # 生成答案
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}

问题：{query}

请提供详细、准确的答案："""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )

        return response.choices[0].message.content

    def chat(self, query: str, history: List[Dict] = None) -> str:
        """对话式RAG"""
        if not self.client:
            return "LLM客户端未配置"

        # 检索
        relevant = self.search(query, limit=3)
        context = "\n\n".join([r['content'] for r in relevant])

        # 构建消息
        messages = [
            {"role": "system", "content": f"你是知识库助手。基于以下上下文回答问题：\n\n{context}"}
        ]

        if history:
            messages.extend(history[-6:])

        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000
        )

        return response.choices[0].message.content

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "documents_count": len(self.documents),
            "vectors_count": self.store.count(),
            "persist_dir": self.persist_dir
        }

    def clear(self):
        """清空所有数据"""
        self.store.clear()
        self.documents.clear()


def create_rag(**kwargs) -> RAGEngine:
    """创建RAG引擎"""
    return RAGEngine(**kwargs)


if __name__ == "__main__":
    rag = create_rag()

    print("RAG Engine")
    print(f"Stats: {rag.get_stats()}")
    print()

    # 测试添加文档
    rag.add_document("Python是一种高级编程语言，广泛用于Web开发、数据科学和人工智能。", {"topic": "Python"})
    rag.add_document("机器学习是人工智能的一个分支，它使计算机能够从数据中学习。", {"topic": "ML"})
    rag.add_document("深度学习使用神经网络来模拟人类大脑的工作方式。", {"topic": "DL"})

    print(f"After adding: {rag.get_stats()}")
    print()

    # 测试搜索
    results = rag.search("Python编程")
    print(f"Search results: {len(results)}")
    for r in results:
        print(f"  - {r['content'][:50]}...")

    # 测试生成
    if OPENAI_AVAILABLE:
        answer = rag.generate("什么是Python？")
        print(f"\nAnswer: {answer}")
