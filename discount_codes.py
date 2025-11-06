DISCOUNT_CODES = {
    "WELCOME10": {
        "code": "WELCOME10",
        "discount_percent": 10,
        "description": "Giảm 10% cho khách hàng mới"
    },
    "LOYAL20": {
        "code": "LOYAL20", 
        "discount_percent": 20,
        "description": "Giảm 20% cho khách hàng thân thiết"
    },
    "SPECIAL30": {
        "code": "SPECIAL30",
        "discount_percent": 30,
        "description": "Giảm 30% cho ngày đặc biệt"
    }
}

def validate_discount_code(code):
    """Kiểm tra mã giảm giá có hợp lệ không"""
    if not code:
        return None
    return DISCOUNT_CODES.get(code.upper())

def calculate_discount(total_amount, discount_info):
    """Tính số tiền được giảm"""
    if not discount_info:
        return 0
    discount_percent = discount_info["discount_percent"]
    discount_amount = (total_amount * discount_percent) / 100
    return int(discount_amount)  # Làm tròn số tiền giảm