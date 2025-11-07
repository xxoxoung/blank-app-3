import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    # 1. Tavily Search Tool 호출하기
    search_tool = TavilySearchResults(
        k=6,
        name="web_search"
    )
    return search_tool

# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    # 2. PDF 로더 초기화 및 문서 불러오기
    all_documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 3. 텍스트를 일정 단위(chunk)로 분할하기
    #    - chunk_size: 한 덩어리의 최대 길이
    #    - chunk_overlap: 덩어리 간 겹치는 부분 길이
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_documents = loader.load_and_split(splitter)

    # 4. 분할된 문서들을 임베딩하여 벡터 DB(FAISS)에 저장하기
        vectorstore = FAISS.from_documents(all_documents, OpenAIEmbeddings())

    # 5. 검색기(retriever) 객체 생성
        retriever = vectorstore.as_retriever()

    # 6. retriever를 LangChain Tool 형태로 변환 -> name은 pdf_search로 지정
        retriever_tool = create_retriever_tool(
            retriever,
            name="pdf_search",
            description="Search for information across all uploaded PDF documents"
        )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        # 7. 여러분의 챗봇에 맞는 system message 작성하기
        "당신은 똑똑한 어시스턴트입니다. 당신은 두가지 도구를 사용할 수 있습니다.\n"
        "`pdf_search` : 업로드된 PDF 문서 안에서 답을 검색하는 도구 입니다.\n"
        "1. 항상 먼저 `pdf_search`를 사용하여 답을 찾으려고 하세요.\n"
        "2. 만약 `pdf_search`에서 관련 답변을 찾지 못했거나 불충분하다면, 그 다음에 `web_search`를 사용하세요.\n"
        "3. 두 도구 모두 답을 제공하지 못한다면, '관련 정보를 찾을 수 없습니다.'라고 답하세요.\n"
        "모든 답변은 집에 있는 강아지가 말하듯이 귀엽고 친근하게 맨 끝에는 🐾을 붙여서 대답하세요.\n"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 8.agent 및 agent_executor 생성하기
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수 (툴 사용 내역 제거)
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # 9. intermediate_steps 통해 사용툴을 출력할 수 있는 코드 완성하기
    # intermediate_steps에서 마지막만 가져오기
    used_tools = []
    for step in result.get("intermediate_steps", []):
        tool_name = step[0].tool
        #obs = step[1]
        #if obs and len(str(obs).strip()) > 30:  # 관찰 결과가 충분히 길 때만 기록
        used_tools.append(tool_name)
    used_tools = list(set(used_tools))

    return f"답변:\n{answer}\n\n 사용된 툴: {', '.join(used_tools) if used_tools else '없음'}"

# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    # 10. 여러분의 챗봇에 맞는 스타일로 변경하기
    st.set_page_config(page_title="타이베이 맛집 마스터", layout="wide", page_icon="🐶")
    st.image('data/dog_cook.png', width=300)
    st.title("타이베이 맛집이 궁금해?🐾")  

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")
        pdf_docs = st.file_uploader("PDF 파일 업로드", accept_multiple_files=True)

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web(), load_pdf_files(pdf_docs)]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("먹고 싶은걸 말해라🐾")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()
