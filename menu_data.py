# menu_data.py
# File này chứa danh sách các món ăn chay và các thực đơn được tạo sẵn với đầy đủ thông tin chi tiết.

# Danh sách các mã (tên thư mục) của món ăn có thể nấu chay.
VEGETARIAN_DISHES = [
    "canh_bau", "canh_bi_do", "canh_rau _muong", "canh_rau_cai", "canh_rong_bien",
    "com_trang", "cu_san", "dau_bap", "dau_hu_sot_ca", "dau_que", "dua_hau",
    "kho_qua", "lagim", "thanh_long",
]

# Từ điển chứa các thực đơn mẫu với đầy đủ thông tin chi tiết của từng món.
PRESET_MENUS = {
    "an_chay": {
        "display_name": "💚 Thực đơn Ăn Chay",
        "description": "Bữa ăn thanh đạm, đủ chất, không chứa thịt động vật.",
        "items": [
            {"id": "com_trang", "name": "Cơm trắng", "unit": "1.5 chén", "calories": 195, "price": 10000},
            {"id": "dau_hu_sot_ca", "name": "Đậu hũ sốt cà chua", "unit": "100g", "calories": 197, "price": 25000},
            {"id": "canh_bi_do", "name": "Canh bí đỏ", "unit": "1 bát", "calories": 180, "price": 10000},
            {"id": "lagim", "name": "Rau củ xào (Lagim)", "unit": "100g", "calories": 100, "price": 10000},
            {"id": "dua_hau", "name": "Dưa hấu", "unit": "100g", "calories": 30, "price": 7000}
        ]
    },
    "tang_can": {
        "display_name": "💪 Thực đơn Tăng cơ, Tăng cân",
        "description": "Bữa ăn giàu protein và năng lượng cho người cần tăng cân.",
        "items": [
            {"id": "com_trang", "name": "Cơm trắng", "unit": "1.5 chén", "calories": 195, "price": 10000},
            {"id": "thit_kho_trung", "name": "Thịt kho trứng", "unit": "130g", "calories": 598, "price": 30000},
            {"id": "suon_nuong", "name": "Sườn nướng", "unit": "130g", "calories": 315, "price": 30000},
            {"id": "canh_rong_bien", "name": "Canh rong biển", "unit": "1 bát", "calories": 100, "price": 10000},
            {"id": "tom", "name": "Tôm luộc/hấp", "unit": "130g", "calories": 129, "price": 30000}
        ]
    },
    "giam_can": {
        "display_name": "🏃 Thực đơn Giảm cân, Giữ dáng",
        "description": "Bữa ăn ít calo nhưng vẫn đủ no nhờ nhiều rau và protein nạc.",
        "items": [
            {"id": "thit_luoc", "name": "Thịt heo luộc", "unit": "130g", "calories": 189, "price": 25000},
            {"id": "canh_rau _muong", "name": "Canh rau muống", "unit": "1 bát", "calories": 40, "price": 7000},
            {"id": "dau_bap", "name": "Đậu bắp luộc", "unit": "100g", "calories": 33, "price": 7000},
            {"id": "com_trang", "name": "Cơm trắng", "unit": "1.5 chén", "calories": 195, "price": 10000},
            {"id": "thanh_long", "name": "Thanh long", "unit": "100g", "calories": 57, "price": 7000}
        ]
    },
    "tiet_kiem": {
        "display_name": "💰 Thực đơn Tiết kiệm",
        "description": "Một bữa ăn đầy đủ các món cơ bản với chi phí hợp lý.",
        "items": [
            {"id": "com_trang", "name": "Cơm trắng", "unit": "1.5 chén", "calories": 195, "price": 10000},
            {"id": "trung_chien", "name": "Trứng chiên thịt", "unit": "130g", "calories": 300, "price": 25000},
            {"id": "canh_rau _muong", "name": "Canh rau muống", "unit": "1 bát", "calories": 40, "price": 7000},
            {"id": "dau_bap", "name": "Đậu bắp luộc", "unit": "100g", "calories": 33, "price": 7000},
            {"id": "dua_hau", "name": "Dưa hấu", "unit": "100g", "calories": 30, "price": 7000}
        ]
    },
    "thinh_soan": {
        "display_name": "🍚 Thực đơn Cơm nhà Thịnh soạn",
        "description": "Bữa ăn đậm chất truyền thống, đầy đủ các vị mặn, ngọt, chua.",
        "items": [
            {"id": "com_trang", "name": "Cơm trắng", "unit": "1.5 chén", "calories": 195, "price": 10000},
            {"id": "thit_kho_trung", "name": "Thịt kho trứng", "unit": "130g", "calories": 598, "price": 30000},
            {"id": "ca_hu_kho", "name": "Cá hú kho", "unit": "1 lát vừa", "calories": 185, "price": 30000},
            {"id": "canh_chua_ca", "name": "Canh chua cá", "unit": "1 bát", "calories": 220, "price": 25000},
            {"id": "dua_hau", "name": "Dưa hấu", "unit": "100g", "calories": 30, "price": 7000}
        ]
    }
}