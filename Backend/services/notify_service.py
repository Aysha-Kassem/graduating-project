import requests

def send_mobile_alert(user_id, message):
    """
    لو عندك Firebase أو Expo ضيف هنا
    """
    print(f"[ALERT for user {user_id}] {message}")

    # مثال إرسال إشعار
    # requests.post("https://expo-notify.com", json={
    #     "user": user_id,
    #     "msg": message
    # })
