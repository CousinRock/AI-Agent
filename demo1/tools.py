import json
import datetime
from crewai.tools import BaseTool
from PyPDF2 import PdfReader

# 读取 PDF
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

class ProgressTool(BaseTool):
    name: str = "Progress Tool"
    description: str = "记录并维护 progress.md，保存智能体每次的回答进度"

    progress_file: str = "progress.md"  # ✅ 定义为字段

    def _run(self, agent_name: str, task_description: str, output: str, clear: bool = False) -> str:
        if clear:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                f.write("# Progress Log\n\n")
            return "progress.md 已清空"

        entry = (
            f"## {datetime.datetime.now().isoformat()}\n"
            f"**Agent**: {agent_name}\n\n"
            f"**Task**: {task_description}\n\n"
            f"**Output**:\n{output}\n\n---\n"
        )

        with open(self.progress_file, "a", encoding="utf-8") as f:
            f.write(entry)

        return f"进度已更新到 {self.progress_file}"


class ConversationLoggerTool(BaseTool):
    name: str = "Conversation Logger Tool"
    description: str = "记录用户每次的问题和AI的回答到 conversation_log.txt"

    log_file: str = "conversation_log.txt"  # ✅ 定义为字段

    def _run(self, user_question: str, ai_response: str) -> str:
        """追加写入用户提问和AI回答"""
        log_entry = f"[{datetime.datetime.now().isoformat()}]\n" \
                    f"用户问题: {user_question}\n" \
                    f"AI回答: {ai_response}\n\n"

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        return f"对话已记录到 {self.log_file}"