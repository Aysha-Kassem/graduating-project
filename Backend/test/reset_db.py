import sys, os
from app.database import engine, Base
from sqlalchemy import text

# =================================================
# Optional: إضافة مسار المشروع للـ imports
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def reset_database():
    with engine.connect() as conn:
        print("[INFO] Dropping all tables...")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        Base.metadata.drop_all(bind=engine)
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        print("[INFO] All tables dropped.")

        print("[INFO] Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("[INFO] All tables created successfully.")

if __name__ == "__main__":
    reset_database()
