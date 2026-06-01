@echo off
title mem0 API Server

echo ========================================================
echo     mem0 + Embedding API Server
echo ========================================================
echo.

cd /d "D:\Claude code\mem0-shared"

echo Starting mem0 API Server on port 7888...
echo.
echo   Context Inject: http://localhost:7888/api/context/inject?project=claude^&query=xxx
echo   Memory Search: http://localhost:7888/api/memory/search?project=claude^&query=xxx
echo   Health Check: http://localhost:7888/health
echo.
echo   Shared Chroma DB: D:\Hermes\second-brain\data\mem0_chroma
echo   User IDs: claude, gsd, hermes (logical isolation)
echo.

python api_server.py
