import json
import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

# 讀取相機 JSON 資料作為 Mock 資料庫
SPECS_PATH = "data/camera_specs.json"

def _load_cameras() -> List[Dict[str, Any]]:
    if os.path.exists(SPECS_PATH):
        with open(SPECS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@tool
def search_cameras(
    max_price: Optional[int] = None,
    brand: Optional[str] = None,
    sensor: Optional[str] = None
) -> str:
    """
    根據預算上限、品牌或感光元件類別來篩選符合條件的相機清單。
    
    Args:
        max_price: 預算上限 (例如: 60000)
        brand: 品牌名稱 (例如: "Sony", "Fujifilm", "Canon", "Nikon", "Panasonic", "Ricoh")
        sensor: 感光元件種類 (例如: "Full-Frame" 或 "APS-C")
    """
    cameras = _load_cameras()
    filtered = []

    for cam in cameras:
        # 價格條件過濾
        if max_price is not None and cam["price"] > max_price:
            continue
        # 品牌條件過濾
        if brand is not None and brand.lower() not in cam["brand"].lower():
            continue
        # 感光元件過濾
        if sensor is not None and sensor.lower() not in cam["sensor"].lower():
            continue
            
        filtered.append(cam)

    if not filtered:
        return "查無符合條件的相機，請嘗試放寬預算或篩選條件。"

    results = []
    for c in filtered:
        results.append(
            f"• [{c['name']}] 品牌: {c['brand']} | 價格: NT${c['price']:,} | 片幅: {c['sensor']} | 重量: {c['weight_g']}g\n"
            f"  簡介: {c['description']}"
        )

    return "\n\n".join(results)


@tool
def book_store_visit(user_name: str, phone: str, camera_name: str, visit_date: str) -> str:
    """
    為顧客預約到實體門市現場體驗與試用相機。
    
    Args:
        user_name: 顧客姓名
        phone: 顧客聯絡電話
        camera_name: 想試用的相機型號
        visit_date: 預約日期與時間 (例如: "2026-10-10 14:00")
    """
    # 這裡模擬寫入數據庫或發送 Notification
    booking_info = {
        "user_name": user_name,
        "phone": phone,
        "camera_name": camera_name,
        "visit_date": visit_date,
        "status": "Confirmed"
    }
    
    # 印出日誌供除錯驗證
    print(f"\n[API Log] 收到門市預約登記：{booking_info}")
    
    return f"✅ 預約成功！已為 {user_name} 先生/小姐登記預約，日期：{visit_date}，預約試用機型：{camera_name}。我們的專員會再透過電話 {phone} 與您確認細節。"


if __name__ == "__main__":
    # 本地快速驗證工具邏輯
    print("🧪 測試工具 1: 搜尋預算 30000 以內且是 APS-C 的相機...")
    print(search_cameras.invoke({"max_price": 30000, "sensor": "APS-C"}))
    
    print("\n" + "="*50 + "\n")
    
    print("🧪 測試工具 2: 門市帶看預約測試...")
    print(book_store_visit.invoke({
        "user_name": "王小明",
        "phone": "0912345678",
        "camera_name": "Sony ZV-E10 II",
        "visit_date": "2026-10-10 15:00"
    }))