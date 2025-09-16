import os
from crewai import Agent, LLM, Task, Crew, Process
from crewai.tools import BaseTool
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 工具：读取 PDF
class PDFReaderTool(BaseTool):
    name: str = "PDF Reader Tool"
    description: str = "用于读取 PDF 文件内容的工具"

    def _run(self, file_path: str) -> str:
        """读取 PDF 文件并返回文本"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text


class PDFCrew:
    def __init__(self):
        # 初始化 LLM
        self.llm = LLM(
            base_url="https://api.deepseek.com",
            api_key=DEEPSEEK_API_KEY,
            model="deepseek/deepseek-chat",
            temperature=0.3, 
            timeout=120,
            max_retries=3
        )

        # 工具
        self.pdf_tool = PDFReaderTool()

        # === 定义多个智能体 ===
        self.reader = Agent(
            role="文档读取员",
            goal="负责读取 PDF 文本并提供原始材料",
            backstory="你擅长快速提取 PDF 文档中的全部文本。",
            tools=[self.pdf_tool],
            llm=self.llm,
            verbose=True
        )

        self.summarizer = Agent(
            role="文档总结专家",
            goal="提取 PDF 的摘要和重点",
            backstory="你擅长从冗长文档中提炼出核心信息。",
            llm=self.llm,
            verbose=True
        )

        self.critic = Agent(
            role="学术评论员",
            goal="指出 PDF 内容的不足和缺陷",
            backstory="你专注于文献批判和逻辑分析，能发现漏洞和不足。",
            llm=self.llm,
            verbose=True
        )

        self.innovator = Agent(
            role="创新顾问",
            goal="提出改进和创新点",
            backstory="你专注于将文档内容转化为新的想法和改进方案。",
            llm=self.llm,
            verbose=True
        )

        self.qna = Agent(
            role="问答助手",
            goal="基于 PDF 回答用户提出的问题",
            backstory="你能结合文档内容精准回答问题。",
            llm=self.llm,
            verbose=True
        )

    # === 任务定义 ===
    def create_tasks(self, file_path: str, user_question: str = None):
        tasks = [
            Task(
                description=f"请读取 PDF 文件 {file_path} 的完整内容。",
                agent=self.reader,
                expected_output="原始 PDF 文本"
            ),
            Task(
                description=f"请总结 PDF 文件 {file_path} 的主要内容，包括：\n1. 文件标题\n2. 摘要\n3. 关键点",
                agent=self.summarizer,
                expected_output="PDF 摘要和关键点"
            ),
            Task(
                description=f"请分析 PDF 文件 {file_path} 的不足，包括逻辑漏洞、缺少论证、数据不足。",
                agent=self.critic,
                expected_output="不足与改进方向"
            ),
            Task(
                description=f"请基于 PDF 文件 {file_path} 的内容，提出可改进和创新的部分。",
                agent=self.innovator,
                expected_output="创新和改进建议"
            )
        ]
        if user_question:
            tasks.append(
                Task(
                    description=f"请基于 PDF 文件 {file_path} 回答问题：{user_question}",
                    agent=self.qna,
                    expected_output="基于 PDF 的问答"
                )
            )
        return tasks

    def analyze_pdf(self, file_path: str, user_question: str = None):
        tasks = self.create_tasks(file_path, user_question)

        crew = Crew(
            agents=[self.reader, self.summarizer, self.critic, self.innovator, self.qna],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff()


def main():
    print("=== PDF 多智能体分析 Crew ===")
    pdf_crew = PDFCrew()

    file_path = "./test.pdf" 
    question = "这份文件的核心结论是什么？"

    try:
        result = pdf_crew.analyze_pdf(file_path, question)
        print("\n=== 最终结果 ===")
        print(result)
    except Exception as e:
        print(f"分析 PDF 时出错：{e}")


if __name__ == "__main__":
    main()
