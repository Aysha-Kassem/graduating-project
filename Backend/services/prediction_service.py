from model_loader import ensemble_predict
from models import Prediction
from services.motion_service import get_last_motion_window
from services.notify_service import send_mobile_alert

def run_prediction(db, user_id):
    window = get_last_motion_window(db, user_id)

    if window is None:
        return None

    result = ensemble_predict(window)

    pred = Prediction(
        user_id=user_id,
        fall_now=result["fall_now"],
        fall_soon=result["fall_soon"]
    )
    db.add(pred)
    db.commit()

    if result["fall_now"] > 0.7:
        send_mobile_alert(user_id, "⚠️ سقوط الآن!")
    elif result["fall_soon"] > 0.7:
        send_mobile_alert(user_id, "⚠️ احتمالية سقوط قريب!")

    return result
