from mcp.server.fastmcp import FastMCP
import random
import datetime

# 初始化MCP服务器
mcp = FastMCP("ChatServer", "对话服务示例", port=8090)


async def ai_chat(query: str) -> str:
    """智能对话工具，处理用户输入并生成回复"""
    responses = [
        "这是一个基于MCP协议的回复示例",
        "您的问题是：" + query,
        "MCP让AI能轻松连接外部工具",
        "正在处理您的请求：" + query[:20] + "...",
        "这是随机生成的回复ID：" + str(random.randint(1000,9999))
    ]
    return random.choice(responses)

@mcp.tool("获取当前时间", "获取指定时区的当前时间")
def get_current_time(timezone: str = "Asia/Shanghai") -> dict:
    """
    获取指定时区的当前时间，默认时区为上海（北京时间）。
    参数:
      timezone (str): IANA标准时区名（如"America/New_York"）
    返回:
      {"time": "YYYY-MM-DD HH:MM:SS", "timezone": str}
    """
    now = datetime.datetime.now().astimezone()  # 自动识别时区
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"time": formatted_time, "timezone": str(now.tzinfo)}

@mcp.tool("获取天气", "获取指定城市的天气")
def get_weather_city(city: str = "北京") -> dict:
    """
    获取指定城市的当前天气，默认北京。
    参数:f
      city (str): 北京
    返回:
      {"weather": "今日$city 天气晴朗", "city": str}
    """
    return {"weather": f"今日{city}天气晴朗", "city": city}

@mcp.tool("数学计算", "执行基本数学运算")
def calculate(expression: str) -> float:
    """计算数学表达式"""
    try:
        return eval(expression)
    except:
        return "计算错误：无效表达式"

if __name__ == '__main__':
    # 启动SSE协议的MCP服务器
    mcp.run(transport='stdio')