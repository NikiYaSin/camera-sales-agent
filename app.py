import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_agent_executor

# 1. 初始化 LangGraph Agent
print("🚀 正在啟動 LangGraph Agent 系統...")
agent_app = build_agent_executor()


def chat_response(user_input, chat_history):
    """
    處理使用者輸入並透過 LangGraph Agent 取得回應
    chat_history 格式:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    if not user_input.strip():
        return "", chat_history

    # 1. 重新建構傳給 Agent 的歷史訊息 (LangChain Message 格式)
    langchain_messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    # 追加最新的使用者訊息
    langchain_messages.append(HumanMessage(content=user_input))

    # 2. 呼叫 Agent 進行圖狀態推進
    initial_state = {"messages": langchain_messages}
    output_state = agent_app.invoke(initial_state)

    # 3. 取得 Agent 最終回應內容
    bot_response = output_state["messages"][-1].content

    # 4. 更新 Gradio UI 的歷史紀錄 (符合 Dict 格式)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_response})

    return "", chat_history


# 2. 打造 Gradio 網頁介面
with gr.Blocks(title="鏡界 (LensGuide) - 相機門市 AI 銷售顧問") as demo:
    gr.Markdown(
        """
        # 📷 鏡界 (LensGuide) 相機門市 AI 銷售顧問
        歡迎來到鏡界門市！我是您的專屬相機選購顧問，能為您提供：
        - 💡 **攝影觀念解答**：全片幅 vs APS-C、Vlog 挑選建議
        - 🔍 **商品庫存篩選**：根據預算、品牌與需求推薦相機
        - 🗓️ **門市帶看預約**：安排親臨實體門市體驗實機
        """
    )

    # 直接宣告 Chatbot 即可 (移除 type 參數)
    chatbot = gr.Chatbot(label="對話歷史", height=500)

    with gr.Row():
        msg_input = gr.Textbox(
            show_label=False,
            placeholder="請輸入您的問題，例如：『想找 60000 以內適合旅遊拍人像的相機』",
            scale=8,
        )
        submit_btn = gr.Button("發送訊息", variant="primary", scale=2)

    with gr.Row():
        clear_btn = gr.ClearButton([msg_input, chatbot], value="🧹 清除對話紀錄")

    # 快速測試按鈕組 (快捷引導)
    gr.Markdown("### 💡 快速發問範例：")
    gr.Examples(
        examples=[
            "vlog挑選建議是什麼意思？",
            "全片幅跟 APS-C 有什麼差別？",
            "我想找預算 30000 以內、適合新手拍 Vlog 的相機",
            "我想預約下週六下午 2 點到門市試用 Sony A74，我叫王小明，電話 0912345678",
        ],
        inputs=msg_input,
    )

    # 事件綁定：按下 Enter 或點擊發送按鈕觸發對話
    msg_input.submit(
        chat_response, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
    )
    submit_btn.click(
        chat_response, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot]
    )

# 3. 啟動應用程式
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)