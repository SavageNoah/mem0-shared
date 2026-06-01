#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes API Server
提供 mem0 记忆搜索接口，供 cc-connect (Claude/GSD) 调用
兼容 claude-mem 的 /api/context/inject 接口格式
"""
import os
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn

# 导入 memory 模块
sys.path.insert(0, os.path.dirname(__file__))
from memory import get_memory

app = FastAPI(title="Shared Memory API", version="1.1")


class AddMemoryRequest(BaseModel):
    content: str
    project: str = "hermes"
    infer: bool = False


def _search_mem0(project: str, query: str, limit: int = 5) -> list:
    """搜索指定项目的记忆"""
    m = get_memory()
    if m is None:
        return []

    # project 映射到 user_id
    # "claude", "gsd", "hermes" 分别搜索各自的记忆
    user_id = project.lower() if project else "hermes"

    try:
        # 尝试带 filters 的方式
        results = m.search(query, filters={"user_id": user_id}, limit=limit)
    except:
        try:
            # 老版本 mem0 的 API
            results = m.search(query, user_id=user_id, limit=limit)
        except Exception as e:
            print(f"[API] 搜索失败: {e}")
            return []

    # 统一返回格式处理
    if isinstance(results, dict) and "results" in results:
        return results["results"]
    if isinstance(results, list):
        return results
    return []


def _format_for_context_inject(results: list) -> str:
    """格式化成 claude-mem 的文本注入格式"""
    if not results:
        return ""

    parts = []
    for r in results:
        mem_text = ""
        if isinstance(r, dict):
            mem_text = r.get("memory", r.get("text", ""))
        elif isinstance(r, str):
            mem_text = r
        else:
            try:
                mem_text = getattr(r, "memory", getattr(r, "text", ""))
            except:
                continue
        if mem_text:
            parts.append(f"- {mem_text}")

    if not parts:
        return ""

    return "\n".join(parts)


@app.get("/api/context/inject", response_class=PlainTextResponse)
async def context_inject(query: str, project: str = "hermes"):
    """
    兼容 claude-mem 的记忆注入接口

    参数:
    - query: 搜索关键词（用户消息内容）
    - project: 项目名，对应 user_id (claude, gsd, hermes)

    返回:
    - 格式化的记忆文本，直接注入到 LLM prompt 中
    """
    try:
        results = _search_mem0(project, query, limit=5)
        formatted = _format_for_context_inject(results)
        return formatted
    except Exception as e:
        print(f"[API] context_inject 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memory/search")
async def search_memory(query: str, project: str = "hermes", limit: int = 5):
    """
    原始记忆搜索接口，返回结构化 JSON
    """
    try:
        results = _search_mem0(project, query, limit)
        return {"project": project, "query": query, "results": results}
    except Exception as e:
        print(f"[API] search_memory 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "shared-memory-api", "embedding": "BAAI/bge-base-zh-v1.5"}


@app.post("/api/memory/add")
async def add_memory(req: AddMemoryRequest):
    """
    写入记忆 (POST body)

    Body:
    - content: 记忆内容
    - project: 项目名，对应 user_id (claude, gsd, hermes)
    - infer: 是否让 mem0 自动提取记忆（默认 False）
    """
    try:
        m = get_memory()
        if m is None:
            raise HTTPException(status_code=503, detail="mem0 未初始化")
        user_id = req.project.lower() if req.project else "hermes"
        m.add(req.content, user_id=user_id, infer=req.infer)
        return {"status": "ok", "project": req.project, "content_length": len(req.content)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_background_tasks():
    """后台运行的任务（文件监控等，以后可以扩展）"""
    # 暂时只跑 API，文件监控可以独立启动
    # 未来可以整合 ingest.py 的功能到这里
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Hermes Memory API Server 启动中...")
    print("=" * 60)
    print(f"  接口地址: http://localhost:7888")
    print(f"  记忆注入: http://localhost:7888/api/context/inject?project=claude&query=xxx")
    print(f"  健康检查: http://localhost:7888/health")
    print("=" * 60)
    print("  支持的 project: claude, gsd, hermes")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7888,
        log_level="warning"
    )
