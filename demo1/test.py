import os
from crewai import Agent, LLM, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, SerperDevTool
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GUI_API_KEY = os.getenv("GUI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class WebContentAgent:
    def __init__(self):
        # 初始化LLM
        # self.llm = LLM(
        #     base_url="https://api.deepseek.com",
        #     api_key=DEEPSEEK_API_KEY,
        #     model="deepseek/deepseek-chat",
        #     temperature=0.3,  # 降低随机性，让 Agent 更专注于任务执行
        #     timeout=120,      # 训练模型任务耗时较长，延长超时时间
        #     max_retries=3
        # )

        self.llm = LLM(
            base_url="https://api.chatanywhere.tech",
            api_key=OPENAI_API_KEY,
            model="openai/gpt-4.1-mini",
            temperature=0.3, 
            timeout=120,     
            max_retries=3
        )

        
        # 创建网页搜索工具
        self.website_tool = SerperDevTool()
        
        # 创建智能体
        self.web_agent = Agent(
            role="网页内容分析师",
            goal="从指定网页中提取和分析内容信息",
            backstory="""你是一个专业的网页内容分析师，擅长从各种网页中提取关键信息。
            你能够理解网页结构，识别重要内容，并以结构化的方式整理信息。""",
            tools=[self.website_tool],
            llm=self.llm,
            verbose=True
        )
    
    def create_web_content_task(self, url: str, specific_info: str = None):
        """创建网页内容获取任务"""
        if specific_info:
            task_description = f"""
            请访问网页 {url}，并提取以下特定信息：{specific_info}
            
            请按以下格式整理信息：
            1. 网页标题
            2. 主要内容摘要
            3. 关键信息点
            4. 相关链接（如果有）
            5. 其他重要信息
            """
        else:
            task_description = f"""
            请访问网页 {url}，分析并提取网页的主要内容。
            
            请按以下格式整理信息：
            1. 网页标题
            2. 主要内容摘要
            3. 关键信息点
            4. 相关链接（如果有）
            5. 其他重要信息
            
            请确保信息的准确性和完整性。
            """
        
        return Task(
            description=task_description,
            agent=self.web_agent,
            expected_output="结构化的网页内容分析报告，包含标题、摘要、关键信息点等"
        )
    
    def get_web_content(self, url: str, specific_info: str = None):
        """获取网页内容"""
        # 创建任务
        task = self.create_web_content_task(url, specific_info)
        
        # 创建Crew
        crew = Crew(
            agents=[self.web_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        # 执行任务
        result = crew.kickoff()
        return result

def main():
    """主函数 - 演示如何使用网页内容获取智能体"""
    print("=== CrewAI 网页内容获取智能体 ===")

    # 创建智能体
    web_agent = WebContentAgent()
    
    # 示例1：获取网页的完整内容
    print("\n示例1：获取网页完整内容")
    url1 = "https://docs.crewai.com/en/learn/llm-connections#using-the-llm-class" 
    try:
        result1 = web_agent.get_web_content(url1)
        print(f"网页 {url1} 的内容分析结果：")
        print(result1)
    except Exception as e:
        print(f"获取网页内容时出错：{e}")
    
    # # 示例2：获取特定信息
    # print("\n示例2：获取特定信息")
    # url2 = "https://zhuanlan.zhihu.com/p/677696671"  # 替换为你想分析的网页
    # specific_info = "最新的技术新闻标题和链接"
    # try:
    #     result2 = web_agent.get_web_content(url2, specific_info)
    #     print(f"从 {url2} 获取的特定信息：")
    #     print(result2)
    # except Exception as e:
    #     print(f"获取特定信息时出错：{e}")

if __name__ == "__main__":
    main()