# 📷 鏡界 (LensGuide) — AI 相機門市銷售顧問 Agent

一個基於 **LangGraph**、**LangChain** 與 **RAG (Retrieval-Augmented Generation)** 的生產級 AI 相機銷售顧問系統，並提供 **Gradio Web UI** 互動介面。

---

## 🌟 專案特色

* **RAG 向量檢索 (ChromaDB)**：針對相機規格與常見攝影知識 (FAQ) 進行精確檢索，防止模型產生幻覺。
* **動態工具調用 (Tool Calling)**：
  * **商品條件過濾 (`search_cameras`)**：依據預算上限、品牌與感光元件規格進行多條件庫存篩選。
  * **門市帶看預約 (`book_store_visit`)**：收集顧客資訊並動態完成實體門市體驗登記。
* **LangGraph 狀態機管理**：建構 `Chatbot -> Tools -> Chatbot` 的循環工作流，並導入角色設定與安全護欄 (Guardrails)。
* **Gradio Web UI 介面**：提供簡潔、響應式的網頁對話介面，並支援歷史對話紀錄與快速發問範例。

---

## 🛠️ 技術棧 (Tech Stack)

* **LLM Core**: OpenAI (`gpt-4o-mini`)
* **Framework**: LangGraph, LangChain (`langchain-core`, `langchain-openai`, `langchain-chroma`)
* **Vector Database**: ChromaDB
* **Frontend UI**: Gradio 6.x
* **Language & Env**: Python 3.13, `python-dotenv`

---

## 📁 專案架構目錄

```text
camera-sales-agent/
├── data/                  # 知識庫原始資料 (JSON)
│   ├── camera_specs.json  # 相機規格資料
│   └── camera_faqs.json   # 攝影觀念 FAQ 資料
├── src/                   # 核心邏輯模組
│   ├── ingest.py          # 向量資料庫建立腳本 (ChromaDB Ingestion)
│   ├── tools.py           # LangChain 工具庫 (庫存查詢、門市預約)
│   ├── prompts.py         # System Prompt 與 Guardrails 指令
│   └── agent.py           # LangGraph 狀態機圖架構
├── app.py                 # Gradio Web UI 主程式
├── requirements.txt       # 專案套件依賴清單
├── .env.example           # 環境變數設定範本
└── README.md              # 專案說明文件
```

---

## 🚀 快速啟動指南

### 1. 複製專案與建立虛擬環境

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd camera-sales-agent

# 建立並啟動虛擬環境
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows
```

### 2. 安裝套件依賴

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數

在專案根目錄下建立 .env 檔案並填入 OpenAI API Key：

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 建立向量資料庫 (Ingestion)

```bash
python src/ingest.py
```

### 5. 啟動 Web UI 服務

```bash
python app.py
```

啟動後請開啟瀏覽器訪問：http://127.0.0.1:7860