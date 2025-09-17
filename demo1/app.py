import streamlit as st
import tempfile
from main import PDFCrew

# 初始化
if "crew" not in st.session_state:
    st.session_state.crew = PDFCrew()
if "messages" not in st.session_state:
    st.session_state.messages = []  # 保存对话历史

pdf_crew = st.session_state.crew

st.set_page_config(page_title="文献助手", layout="wide")
st.title("文献助手")

# === 文件上传 ===
uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.pdf_path = tmp.name

     # ✅ 上传后立即进行预处理（缓存内容）并显示等待提示
    with st.spinner("正在读取和解析 PDF，请稍候..."):
        pdf_crew.preprocess_pdf(st.session_state.pdf_path)

    st.success("✅ PDF 上传完成，可以开始提问啦！")

# === 聊天 UI ===
if "pdf_path" in st.session_state:
    # 展示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if user_input := st.chat_input("请输入你的问题"):
        # 显示用户输入
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 智能体回答
        with st.chat_message("assistant"):
            with st.spinner("智能体正在思考中..."):
                results = pdf_crew.analyze_pdf(st.session_state.pdf_path, user_input)
                answer = results[-1] if results else "没有得到回答"
                st.markdown(answer)

        # 保存到对话历史
        st.session_state.messages.append({"role": "assistant", "content": answer})
