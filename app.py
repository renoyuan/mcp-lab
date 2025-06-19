import asyncio
import re
import json
from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from openai import OpenAI
from openai import AsyncOpenAI 

import asyncio
import json

from openai import OpenAI,AsyncOpenAI

from fastapi import FastAPI, Request, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters

import tomllib as tomli  # Python < 3.11
# 或使用内置库（Python ≥ 3.11）: import tomllib as tomli

with open("config.toml", "rb") as f:  # 二进制模式避免编码问题
    cfg = tomli.load(f)
    
base_url = cfg["global"]["base_url"] 
api_key = cfg["global"]["api_key"] 
ollama_llm = OpenAI(
    base_url=base_url,  # 修正Ollama API路径
    api_key=api_key  # 认证占位符
)
# 本地加载 mcp-tools
server_params =StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)





app = FastAPI()
templates = Jinja2Templates(directory="templates")

# MCP服务器配置
MCP_SERVER_URL = "http://localhost:8090/sse"

def build_messages(query: str, tool_results: list[dict]) -> list[dict]:
    """
    构建支持多工具调用、动态上下文的消息结构
    :param query: 原始用户问题
    :param tool_results: 工具调用结果列表，格式 [{"tool_name": str, "content": str}]
    :return: 符合OpenAI格式的消息列表
    """
    # 1. 聚合工具结果
    result_content = "\n\n".join(
        f"【{res['tool_name']}】\n{res['content']}" 
        for res in tool_results
    )
    
    # 2. 构建系统指令模板
    system_directive = (
        "你是一个专业助手，需严格遵循：\n"
        "1. 直接基于工具结果回答问题\n"
        "2. 禁止展示思考过程\n"
        "3. 当工具结果冲突时标注矛盾点\n"
        "---工具返回结果---\n"
        f"{result_content}"
    )
    
    # 3. 用户指令强化
    user_prompt = (
        f"问题：{query}\n"
        "要求：\n"
        "- 用简洁的陈述句回答\n"
        "- 如信息不足请明确说明\n"
        "/no_think"  # 强化指令标记
    )
    
    return [
        {"role": "system", "content": system_directive},
        {"role": "user", "content": user_prompt}
    ]
    
async def llm_driven_tool_call(query: str):
    # 1. 获取服务端工具列表
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话连接
            await session.initialize()
            print("🔌 MCP会话连接成功")
            
            # 加载工具
            response = await session.list_tools()  # ⚠️ 注意 await
            tools = response.tools  # 提取工具对象列表
            print("可用工具:", [tool.name for tool in tools])  # 打印工具名称[1,3](@ref)
            tools_params = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema  # 使用 JSON Schema 约束参数
                    }
                } for tool in tools]
            print("tools_params", tools_params)
            
            # 2. LLM 决策调用哪个工具及参数
            try:
                llm_response = ollama_llm.chat.completions.create(
                    model="qwen3:4b",  # 移除模型名称后的空格
                    temperature=0.1,
                    messages=[{"role": "user", "content": f"{query}" }],
                    stream=False,  # 禁用流式输出
                    max_tokens=4096,              # 上下文长度
                    tools=tools_params
                )
                
                print("llm_response",llm_response)
                
                # 3. 解析 LLM 的调用指令
                tool_calls = llm_response.choices[0].message.tool_calls
                tools_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    print(f"调用工具: {tool_name}, 参数: {tool_args}")
                    
                    # 4. 执行工具调用并收集结果
                    result = await session.call_tool(tool_name, tool_args)
                    tools_results.append({"tool_name": tool_name, "content": result.content[0].text})  # 存储每个工具的结果

                print("所有工具调用结果:", tools_results)
                messages = build_messages(query,tools_results)
            except Exception as e:
                print(str(e)) 
                tools_results = []
                messages = build_messages(query,tools_results)
            # 5. 将结果返回给 LLM 生成最终回答f
            llm_response = ollama_llm.chat.completions.create(
                model="qwen3:4b",
                max_tokens=4096,
                messages=messages,
            )
            return llm_response.choices[0].message.content
        
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """渲染聊天界面"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket通信处理"""
    await websocket.accept()
    while True:
        # 接收用户消息
        # print("client",client)
        user_input = await websocket.receive_text()
        
        # 调用MCP工具
        # Broken Code (tuple error)
        response = await llm_driven_tool_call(user_input)
        # response = await client.call_tool("ai_chat",parameters={"query": user_input})
        
        
        pattern = r'<think>.*?</think>'
        response = re.sub(pattern, '', response, flags=re.DOTALL)
        
        # todo 发送MCP响应 构建mcp 
        print("AI回复",response)
        await websocket.send_text(f"AI回复: {response}")


from uvicorn import Server
class ProactorServer(Server):
    def run(self, sockets=None):
        loop = asyncio.ProactorEventLoop()  # 显式指定事件循环
        asyncio.set_event_loop(loop)
        super().run(sockets=sockets)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000) # linux 
    config = uvicorn.Config("app:app", host="0.0.0.0", port=8000, reload=False)
    server = ProactorServer(config)
    server.run()