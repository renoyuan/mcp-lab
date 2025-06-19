# mcp-lab

research mcp

一个mcp 实验

# 系统架构设计

1. **前端**：FastAPI + Jinja2模板的Web界面
2. **后端**：FastMCP实现的MCP服务

配置 环境文件

mv config.toml.bak config.toml  写入llm api 信息

# 启动

## 终端1：启动MCP服务

python mcp_server.py

## 终端2：启动前端应用

uvicorn app:app --reload
