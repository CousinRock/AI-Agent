import os
import json
import datetime
from crewai import Agent, LLM, Task, Crew, Process
from dotenv import load_dotenv
from tools import *


# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


class PDFCrew:
    def __init__(self):
        self.memory_file = "progress.md"
        self.pdf_cache = None  # ✅ 缓存 PDF 文本
        self.summary_cache = None  # ✅ 缓存摘要
        self.analysis_cache = None  # ✅ 缓存批判 & 创新

        # 初始化 LLM
        self.llm1 = LLM(
            base_url="https://api.deepseek.com",
            api_key=OPENAI_API_KEY ,
            model="openai/gpt-oss-20b",
            temperature=0.3,
            timeout=120,
            max_retries=3
        )

        self.llm2 = LLM(
            base_url="https://api.chatanywhere.tech",
            api_key=OPENAI_API_KEY,
            model="openai/gpt-4.1-nano",
            temperature=0.3,
            timeout=120,
            max_retries=3
        )

        # 工具
        self.pdf_tool = PDFReaderTool()
        self.progress_tool = ProgressTool()
        self.logger_tool = ConversationLoggerTool()

        # === 定义多个智能体 ===
        self.reader = Agent(
            role="文档读取员",
            goal="负责读取 PDF 文本并提供原始材料",
            backstory="你擅长快速提取 PDF 文档中的全部文本。",
            tools=[self.pdf_tool],
            llm=self.llm1,
            verbose=True
        )

        self.summarizer = Agent(
            role="文档总结专家",
            goal="提取 PDF 的摘要和重点",
            backstory="你擅长从冗长文档中提炼出核心信息。",
            llm=self.llm1,
            verbose=True
        )

        self.critic = Agent(
            role="学术评论员",
            goal="指出 PDF 内容的不足和缺陷",
            backstory="你专注于文献批判和逻辑分析，能发现漏洞和不足。",
            llm=self.llm1,
            verbose=True
        )

        self.innovator = Agent(
            role="创新顾问",
            goal="提出改进和创新点",
            backstory="你专注于将文档内容转化为新的想法和改进方案。",
            llm=self.llm2,
            verbose=True
        )

        self.qna = Agent(
            role="问答助手",
            goal="基于 PDF 回答用户提出的问题",
            backstory="你能结合文档内容精准回答问题。",
            llm=self.llm2,
            verbose=True
        )

    # === 任务定义 ===
    def create_tasks(self, file_path: str, user_question: str = None, memory: str = ""):
        tasks = [
            Task(
                description=f"请读取 PDF 文件 {file_path} 的完整内容。\n\n【历史对话记录】\n{memory}",
                agent=self.reader,
                expected_output="原始 PDF 文本"
            ),
            Task(
                description=f"请总结 PDF 文件 {file_path} 的主要内容。\n\n【历史对话记录】\n{memory}",
                agent=self.summarizer,
                expected_output="PDF 摘要和关键点"
            ),
            Task(
                description=f"请分析 PDF 文件 {file_path} 的不足。\n\n【历史对话记录】\n{memory}",
                agent=self.critic,
                expected_output="不足与改进方向"
            ),
            Task(
                description=f"请基于 PDF 文件 {file_path} 提出创新点。\n\n【历史对话记录】\n{memory}",
                agent=self.innovator,
                expected_output="创新和改进建议"
            )
        ]
        if user_question:
            tasks.append(
                Task(
                    description=f"请基于 PDF 文件 {file_path}\
                          回答问题：{user_question}\n\n【历史对话记录】\n{memory}",
                    agent=self.qna,
                    expected_output="基于 PDF 的问答"
                )
            )
        return tasks
    
    def load_memory(self):
        """读取历史对话作为上下文"""
        if not os.path.exists(self.memory_file):
            return ""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            return f.read()

    def clear_memory(self):
        """清空记忆"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write("")

    def preprocess_pdf(self, file_path: str):
        """只运行一次的预处理"""
        if self.pdf_cache is not None:
            return  # ✅ 已经处理过，跳过

        tasks = [
            Task(description=f"请读取 PDF 文件 {file_path} 的完整内容。",
                 agent=self.reader,
                 expected_output="原始 PDF 文本"),
            Task(description=f"请总结 PDF 文件 {file_path} 的主要内容。",
                 agent=self.summarizer,
                 expected_output="PDF 摘要和关键点"),
            Task(description=f"请分析 PDF 文件 {file_path} 的不足。",
                 agent=self.critic,
                 expected_output="不足与改进方向"),
            Task(description=f"请基于 PDF 文件 {file_path} 提出创新点。",
                 agent=self.innovator,
                 expected_output="创新和改进建议")
        ]

        crew = Crew(
            agents=[self.reader, self.summarizer, self.critic, self.innovator],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        results = crew.kickoff()

        # ✅ 从 tasks_output 里取出结果
        task_outputs = [t.raw for t in results.tasks_output]

        self.pdf_cache = task_outputs[0]
        self.summary_cache = task_outputs[1]
        self.analysis_cache = {"critic": task_outputs[2], "innovator": task_outputs[3]}

        # ✅ 保存到 progress.md
        for task, output in zip(tasks, results):
            self.progress_tool._run(task.agent.role, task.description, str(output))

    def analyze_pdf(self, file_path: str, user_question: str = None):
        # 先确保 PDF 已处理
        self.preprocess_pdf(file_path)

        memory = self.load_memory()

        results = []
        if user_question:
            # ✅ 只跑 QnA
            task = Task(
                description=f"请基于以下内容回答问题：{user_question}\n\n"
                            f"【文档摘要】\n{self.summary_cache}\n\n"
                            f"【原始内容】\n(部分截取){self.pdf_cache}...\n\n"
                            f"【历史对话记录】\n{memory}",
                agent=self.qna,
                expected_output="基于 PDF 的问答"
            )
            crew = Crew(
                agents=[self.qna],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            results = crew.kickoff()
            answer = results.tasks_output[0].raw  # 这里只有一个 QnA 任务
            self.logger_tool._run(user_question, answer)
            results = [answer]

        return results


def main():
    pdf_crew = PDFCrew()
    file_path = "./test.pdf"

    while True:
        question = input("\n请输入你的问题 (输入 exit 结束对话): ")
        if question.lower() == "exit":
            print("清空记忆并退出。")
            pdf_crew.clear_memory()
            break

        result = pdf_crew.analyze_pdf(file_path, question)
        print("\n=== AI 回答 ===")
        print(result[-1])


if __name__ == "__main__":
    main()
