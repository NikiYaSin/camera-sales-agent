import os
from typing import TypedDict, Annotated, List, Union
from dotenv import load_dotenv

# LangChain & LangGraph 模組
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 載入專案內部模組
from src.prompts import SYSTEM_PROMPT
from src.tools import search_cameras, book_store_visit

load_dotenv()

# =====================================================================
# 1. 初始化 RAG (ChromaDB) 檢索工具
# =====================================================================
def get_rag_tool():
    """將 Chroma 向量資料庫封裝為 LangChain Tool"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./chroma_db"
    
    # 載入持久化的 Chroma 資料庫
    vectorstore = Chroma(
        collection_name="camera_knowledge",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    @tool
    def search_camera_knowledge_base(query: str) -> str:
        """
        當需要查詢相機規格詳情、產品介紹或攝影常識 FAQ (例如: 全片幅與 APS-C 的差別、Vlog相機推薦重點) 時，呼叫此工具。
        """
        docs = retriever.invoke(query)
        if not docs:
            return "知識庫中找不到相關資料。"
        
        results = []
        for doc in docs:
            results.append(f"[參考資料]: {doc.page_content}")
        return "\n\n".join(results)

    return search_camera_knowledge_base


# =====================================================================
# 2. 定義 Agent 狀態 (State)
# =====================================================================
class AgentState(TypedDict):
    # add_messages 會自動處理對話歷史訊息的追加
    messages: Annotated[List[BaseMessage], add_messages]


# =====================================================================
# 3. 建立 LangGraph 工作流
# =====================================================================
def build_agent_executor():
    # 組合所有工具：RAG 工具 + 庫存查詢 + 門市預約
    rag_tool = get_rag_tool()
    tools = [rag_tool, search_cameras, book_store_visit]

    # 初始化 OpenAI LLM，並綁定工具
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    # ---------------- 節點定義 ----------------
    def chatbot_node(state: AgentState):
        """主代理節點：負責理解對話並決定是否呼叫工具"""
        messages = state["messages"]
        
        # 確保對話最前頭帶有 System Prompt
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 工具執行節點
    tool_node = ToolNode(tools=tools)

    # ---------------- 構建 Graph ----------------
    workflow = StateGraph(AgentState)

    # 新增節點
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    # 設定起始點
    workflow.set_entry_point("chatbot")

    # 設定條件邊 (Conditional Edge)
    # 如果 LLM 產出 Tool Call 請求，轉到 "tools" 節點，否則結束流向 END
    workflow.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    # 工具執行完後，回到 "chatbot" 節點整理並回答使用者
    workflow.add_edge("tools", "chatbot")

    # 編譯圖為可執行物件
    app = workflow.compile()
    return app


# =====================================================================
# 4. 本地測試 (Local CLI Test)
# =====================================================================
if __name__ == "__main__":
    print("🚀 啟動 LangGraph Agent 測試介面 (輸入 'q' 離開)...\n")
    agent = build_agent_executor()

    # 初始化對話狀態
    state = {"messages": []}

    while True:
        user_input = input("👤 使用者: ")
        if user_input.lower() in ["q", "exit", "quit"]:
            print("👋 對話結束！")
            break

        # 追加使用者輸入
        state["messages"].append(HumanMessage(content=user_input))

        # 執行 Agent 工作流
        output_state = agent.invoke(state)
        state = output_state  # 更新完整狀態

        # 印出最後一筆 Agent 的回應
        last_message = output_state["messages"][-1]
        print(f"\n🤖 鏡界顧問: {last_message.content}\n")
        print("-" * 50)