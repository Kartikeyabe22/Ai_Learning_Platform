import sqlite3
from datetime import datetime

# Database configuration
DB_PATH = "chat_sessions.db"

def get_db_conn():
    """Get a database connection with row factory set to Row."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database tables."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT
                    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS session_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    file_name TEXT,
                    local_path TEXT,
                    uploaded_at TEXT
                    )""")
    conn.commit()
    return conn

def get_sessions_from_db():
    """Get all session IDs from the database."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT session_id FROM sessions ORDER BY rowid")
    return [row[0] for row in cur.fetchall()]

def add_session_to_db(session_id: str):
    """Add a new session to the database."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session_id, datetime.utcnow().isoformat()))
    conn.commit()

def delete_session_from_db(session_id: str):
    """Delete a session and all its related data from the database."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
    conn.commit()

def add_history_to_db(session_id: str, role: str, content: str):
    """Add a message to the chat history."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.utcnow().isoformat()))
    conn.commit()

def get_history_from_db(session_id: str):
    """Get chat history for a session."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content, timestamp FROM history WHERE session_id = ? ORDER BY id", (session_id,))
    return cur.fetchall()

def add_file_to_db(session_id: str, file_name: str, local_path: str):
    """Add uploaded file metadata to the database."""
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO session_files (session_id, file_name, local_path, uploaded_at) VALUES (?, ?, ?, ?)",
                (session_id, file_name, local_path, datetime.utcnow().isoformat()))
    conn.commit()