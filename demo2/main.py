import os
import json
import datetime
from crewai import Agent, LLM, Task, Crew, Process
from dotenv import load_dotenv
from tools import *


# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EnglishCrew:
    def __init__(self):
        self.memory_file = "progress.md"
        self.eng_cache = None  # ✅ 缓存文本
        self.summary_cache = None  # ✅ 缓存摘要
        self.analysis_cache = None  # ✅ 缓存批判 & 创新