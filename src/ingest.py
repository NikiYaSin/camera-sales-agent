import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def build_vector_db():
    print("🚀 開始讀取相機資料與 FAQ...")
    
    documents = []

    # 1. 讀取相機規格資料
    if os.path.exists("data/camera_specs.json"):
        with open("data/camera_specs.json", "r", encoding="utf-8") as f:
            cameras = json.load(f)
            for cam in cameras:
                content = (
                    f"相機名稱: {cam['name']}\n"
                    f"品牌: {cam['brand']}\n"
                    f"感光元件: {cam['sensor']}\n"
                    f"適合族群: {cam['target_user']}\n"
                    f"產品介紹: {cam['description']}"
                )
                metadata = {
                    "source": "camera_spec",
                    "camera_id": cam["id"],
                    "name": cam["name"],
                    "brand": cam["brand"],
                    "price": cam["price"],
                    "sensor": cam["sensor"],
                    "weight_g": cam["weight_g"]
                }
                documents.append(Document(page_content=content, metadata=metadata))

    # 2. 讀取 FAQ 資料
    if os.path.exists("data/camera_faqs.json"):
        with open("data/camera_faqs.json", "r", encoding="utf-8") as f:
            faqs = json.load(f)
            for faq in faqs:
                content = f"問題: {faq['question']}\n解答: {faq['answer']}"
                metadata = {
                    "source": "camera_faq",
                    "faq_id": faq["id"]
                }
                documents.append(Document(page_content=content, metadata=metadata))

    print(f"📦 共讀取 {len(documents)} 筆文件，準備建立 Vector DB...")

    # 3. 初始化 Embedding 與寫入 Chroma
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_directory = "./chroma_db"

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="camera_knowledge",
        persist_directory=persist_directory
    )
    
    print("✅ Chroma 向量資料庫建立成功！已儲存於 ./chroma_db")

    # 4. 驗證搜尋效果
    print("\n🔍 測試檢索 1：搜尋「預算不高且適合新手拍 Vlog 的相機」...")
    res1 = vectorstore.similarity_search("預算不高且適合新手拍 Vlog 的相機", k=1)
    print(res1[0].page_content)

    print("\n🔍 測試檢索 2：搜尋「全片幅跟 APS-C 差在哪裡」...")
    res2 = vectorstore.similarity_search("全片幅跟 APS-C 差在哪裡", k=1)
    print(res2[0].page_content)

if __name__ == "__main__":
    build_vector_db()