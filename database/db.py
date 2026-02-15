import sqlite3

DB_PATH = "database/predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        detected_class TEXT,
        confidence REAL,
        model_version TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def insert_prediction(filename, detected_class, confidence, model_version):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions (filename, detected_class, confidence, model_version)
    VALUES (?, ?, ?, ?)
    """, (filename, detected_class, confidence, model_version))

    conn.commit()
    conn.close()
