import streamlit as st
import numpy as np
import cv2 # Sử dụng OpenCV
import os
import json
import re
from collections import Counter
import io
import requests

# Import các file xử lý của bạn
import tes1
import image_processor
from food_database import FOOD_DATABASE
from menu_data import PRESET_MENUS
from discount_codes import validate_discount_code, calculate_discount

# --- CẤU HÌNH & KHỞI TẠO ---
CUSTOMER_DATA_PATH = "khachhang"
os.makedirs(CUSTOMER_DATA_PATH, exist_ok=True)
st.set_page_config(page_title="Canteen Thông minh", page_icon="🍽️", layout="wide")

# --- DỮ LIỆU & BIẾN TOÀN CỤC ---
food_ids = [None] + list(FOOD_DATABASE.keys())
food_id_to_index = {fid: i for i, fid in enumerate(food_ids)}
def food_label(fid): return "--- Trống ---" if fid is None else FOOD_DATABASE.get(fid, {}).get("name", str(fid))

# --- KHỞI TẠO SESSION STATE ---
def init_session_state():
    defaults = {
        'mode': 'manual', 'current_user': None, 'user_name_input': "",
        'manual_meal': [None] * 5, 'ai_meal': [None] * 5, 'confirmed_items': [],
        'user_suggestion': {"type": None, "data": []}, 'image_to_process': None,
        'analysis_done': False, 'meal_rating': 'Bình thường'
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init_session_state()

# --- HÀM TIỆN ÍCH & XỬ LÝ DỮ LIỆU ---
def sanitize_filename(name):
    return re.sub(r'[^a-z0-9_]', '', name.lower().replace(" ", "_"))

def save_order_history(username, order_info):
    if not username or not order_info: return False
    try:
        filepath = os.path.join(CUSTOMER_DATA_PATH, f"{sanitize_filename(username)}.json")
        data = {"name": username, "orders": []}
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
        data["orders"].append(order_info)
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu đơn hàng: {e}"); return False

def analyze_preferences(user_data):
    orders = user_data.get("orders", [])
    if not orders: return {"type": None, "data": []}
    if len(orders) < 3: return {"type": "Bữa ăn gần nhất", "data": orders[-1].get("items", [])}
    all_items = [item for order in orders for item in order.get("items", [])]
    return {"type": "Món bạn hay ăn nhất", "data": [item for item, _ in Counter(all_items).most_common(5)]}

def load_user_data():
    username = st.session_state.user_name_input.strip()
    if not username: return
    st.session_state.current_user = username
    filepath = os.path.join(CUSTOMER_DATA_PATH, f"{sanitize_filename(username)}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
        st.session_state.user_suggestion = analyze_preferences(data)
        st.success(f"Chào mừng {username} quay trở lại!")
    else:
        st.session_state.user_suggestion = {"type": "khách mới", "data": []}
        st.info(f"Chào mừng khách hàng mới, {username}!")

# --- HÀM CALLBACK (GỌI KHI CÓ SỰ KIỆN) ---
def start_new_session(): st.session_state.clear(); init_session_state(); st.rerun()
def set_mode(mode): st.session_state.mode = mode

def _update_manual_selections(items):
    """Hàm phụ trợ để cập nhật các lựa chọn món thủ công."""
    new_list = (items + [None] * 5)[:5]
    st.session_state.manual_meal = new_list
    st.session_state.confirmed_items = []
    for i, item_id in enumerate(new_list): st.session_state[f"manual_{i}"] = item_id

def handle_preset_change():
    preset_key = st.session_state.preset_selector
    items = [item["id"] for item in PRESET_MENUS[preset_key]["items"]] if preset_key and preset_key in PRESET_MENUS else []
    _update_manual_selections(items)

def apply_suggestion():
    _update_manual_selections(st.session_state.user_suggestion.get("data", []))
    st.session_state.mode = "manual"

def confirm_meal():
    meal_key = f"{st.session_state.mode}_meal"
    items = [item for item in st.session_state[meal_key] if item]
    if items:
        st.session_state.confirmed_items = items
        st.success("Đã xác nhận món ăn!")
    else:
        st.warning("Vui lòng chọn ít nhất một món!"); st.session_state.confirmed_items = []

def process_payment(order_details):
    if save_order_history(st.session_state.current_user, order_details):
        st.success(f"Đã lưu bữa ăn và đánh giá cho {st.session_state.current_user}!")
        st.balloons()
        start_new_session()

# --- HÀM XỬ LÝ LÕI ---
def run_ai_analysis(image_input):
    with st.spinner("🤖 AI đang phân tích..."):
        try:
            file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            final_img, _ = image_processor.crop_to_4_3(image_bgr)
            h, w, _ = final_img.shape
            rects = [
                (0, 0, int(w*0.6), int(h*0.55)), (int(w*0.6), 0, int(w*0.4), int(h*0.55)),
                (0, int(h*0.55), w//3, int(h*0.45)), (w//3, int(h*0.55), w//3, int(h*0.45)),
                (w//3*2, int(h*0.55), w - (w//3*2), int(h*0.45))
            ]
            results = [tes1.execute(final_img[y:y+h_r, x:x+w_r]).get("final_prediction") for x, y, w_r, h_r in rects]
            st.session_state.ai_meal = results
            for i, res in enumerate(results): st.session_state[f"ai_{i}"] = res
            st.session_state.analysis_done = True
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {e}"); st.session_state.analysis_done = False

def generate_vietqr_b64(amount, description="Thanh toan Canteen"):
    try:
        url = f"https://img.vietqr.io/image/BIDV-0933400555-compact.png?amount={amount}&addInfo={description.replace(' ', '%20')}"
        resp = requests.get(url)
        return io.BytesIO(resp.content) if resp.status_code == 200 else st.error("Không thể tải ảnh QR VietQR.")
    except Exception as e:
        st.error(f"Lỗi tạo QR VietQR: {e}"); return None

# --- CÁC HÀM HIỂN THỊ GIAO DIỆN ---
def render_sidebar():
    with st.sidebar:
        st.title("👨‍🍳 Bảng điều khiển")
        st.divider()
        st.text_input("Nhập tên của bạn:", key="user_name_input", on_change=load_user_data)
        st.button("🔄 Khách mới", on_click=start_new_session, use_container_width=True)
        suggestion = st.session_state.get('user_suggestion', {})
        if st.session_state.current_user and suggestion.get("data"):
            with st.expander(f"✨ Gợi ý: {suggestion['type']}", expanded=True):
                names = [FOOD_DATABASE[i]["name"] for i in suggestion["data"] if i in FOOD_DATABASE]
                st.info(", ".join(names))
                st.button("Áp dụng gợi ý này", on_click=apply_suggestion, use_container_width=True)

def render_meal_selection(mode):
    st.subheader(f"Chọn món {'thủ công' if mode == 'manual' else 'bằng AI'}", divider='blue')
    if mode == 'manual':
        preset_keys = [None] + list(PRESET_MENUS.keys())
        st.selectbox("Chọn thực đơn có sẵn:", options=preset_keys,
            format_func=lambda k: PRESET_MENUS[k]["display_name"] if k else "--- Chọn nhanh ---",
            key="preset_selector", on_change=handle_preset_change)
        st.write("---")
    
    for i in range(5):
        st.session_state[f"{mode}_meal"][i] = st.selectbox(
            f"Món #{i+1}", options=food_ids,
            index=food_id_to_index.get(st.session_state.get(f"{mode}_{i}"), 0),
            format_func=food_label, key=f"{mode}_{i}"
        )
    st.button(f"✅ Xác nhận {'món đã chọn' if mode == 'manual' else 'kết quả AI'}", type="primary", use_container_width=True, on_click=confirm_meal)

def render_ai_mode_inputs():
    st.subheader("Phân tích khay cơm bằng AI", divider='blue')
    col1, col2 = st.columns(2)
    image_input = col1.file_uploader("📂 Tải ảnh", key="uploader") or col2.camera_input("📷 Chụp ảnh", key="camera")
    
    if image_input and image_input != st.session_state.image_to_process:
        st.session_state.image_to_process = image_input
        st.session_state.analysis_done = False
        
    if st.session_state.image_to_process:
        st.image(st.session_state.image_to_process, caption="Ảnh sẽ được phân tích", use_container_width=True)
        if not st.session_state.analysis_done:
            st.button("🔍 Bắt đầu phân tích", type="primary", use_container_width=True, on_click=run_ai_analysis, args=(st.session_state.image_to_process,))
    
    if st.session_state.analysis_done:
        st.write("---"); st.write("Kết quả từ AI (có thể chỉnh sửa):")
        render_meal_selection('ai')

def render_bill():
    if not st.session_state.confirmed_items:
        st.info("👉 Vui lòng chọn món và xác nhận để xem hóa đơn."); return

    st.markdown("<div class='bill-title'>Hóa đơn Tạm tính</div>", unsafe_allow_html=True)
    total_price, total_cal = 0, 0
    for fid in st.session_state.confirmed_items:
        if fid in FOOD_DATABASE:
            item = FOOD_DATABASE[fid]
            total_price += item["price"]; total_cal += item["calories"]
            c1, c2, c3 = st.columns([2.5, 1.5, 1])
            c1.markdown(f"<div class='item-name'>{item['name']}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='price-text'>{item['price']:,} VNĐ</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='calorie-text'>{item['calories']} kcal</div>", unsafe_allow_html=True)
            
    st.divider()
    code = st.text_input("Mã giảm giá:")
    info = validate_discount_code(code) if code else None
    discount = calculate_discount(total_price, info) if info else 0
    final_price = total_price - discount

    if info: st.markdown(f"<div class='discount-text'>✅ Giảm {info.get('discount_percent', 0)}%: -{discount:,} VNĐ</div>", unsafe_allow_html=True)
    elif code: st.error("❌ Mã giảm giá không hợp lệ!")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Tạm tính", f"{total_price:,} VNĐ"); col2.metric("Tổng Lượng Calo", f"~ {total_cal:,} kcal")
    st.markdown("---"); st.markdown(f"<div class='final-price'><h2>Thành tiền: {final_price:,} VNĐ</h2></div>", unsafe_allow_html=True); st.divider()

    st.radio("⭐ Đánh giá chất lượng bữa ăn:", ['Chưa tốt', 'Bình thường', 'Rất tốt'], index=1, key='meal_rating', horizontal=True)
    st.markdown("---")
    
    order_details = {"items": st.session_state.confirmed_items, "total_price": total_price, "final_price": final_price, "total_calories": total_cal, "rating": st.session_state.meal_rating}
    
    pay_col1, pay_col2 = st.columns(2, gap="small")
    with pay_col1:
        if st.button("Thanh toán Tiền mặt 💵", use_container_width=True): process_payment(order_details)
    with pay_col2:
        with st.popover("Thanh toán Ví điện tử 💳", use_container_width=True):
            st.markdown("#### Quét mã QR VietQR để thanh toán (BIDV)")
            qr_buf = generate_vietqr_b64(final_price, "Thanh toan Canteen")
            if qr_buf:
                st.image(qr_buf, caption=f"Số tiền: {final_price:,} VNĐ", use_container_width=True)
                if st.button("✅ Tôi đã chuyển khoản", type="primary"): process_payment(order_details)

# --- CHƯƠNG TRÌNH CHÍNH ---
def main():
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # CSS được ẩn đi cho gọn
    render_sidebar()
    st.title("🍽️ Hóa đơn & Thanh toán")

    if not st.session_state.current_user:
        st.info("👋 Vui lòng nhập tên của bạn ở thanh bên trái để bắt đầu."); st.stop()

    left, right = st.columns([3, 2], gap="large")
    with left:
        c1, c2 = st.columns(2)
        c1.button("✍️ Thủ công", on_click=set_mode, args=("manual",), type="primary" if st.session_state.mode == "manual" else "secondary", use_container_width=True)
        c2.button("🤖 AI", on_click=set_mode, args=("ai",), type="primary" if st.session_state.mode == "ai" else "secondary", use_container_width=True)
        st.divider()
        if st.session_state.mode == "manual": render_meal_selection('manual')
        else: render_ai_mode_inputs()
    with right:
        render_bill()

if __name__ == "__main__":
    main()