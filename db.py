"""
Database abstraction layer.
Supports both local SQLite (development) and Google Cloud Firestore (production).
Set environment variable DB_BACKEND=firestore to use Firestore, otherwise defaults to sqlite.
"""
import os
import json
import sqlite3

DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite").lower()

FieldFilter = None
if DB_BACKEND == "firestore":
    from google.cloud.firestore_v1.base_query import FieldFilter
DB_NAME = os.environ.get("DB_NAME", "research_demo.db")

# Lazy Firestore client
_firestore_client = None


def _get_firestore():
    global _firestore_client
    if _firestore_client is None:
        from google.cloud import firestore
        _firestore_client = firestore.Client()
    return _firestore_client


# ──────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────
def init_db():
    if DB_BACKEND == "firestore":
        return  # Firestore is schemaless, no init needed
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT, linked_researcher TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS progress (username TEXT, persona TEXT, PRIMARY KEY(username, persona))')
    c.execute('CREATE TABLE IF NOT EXISTS surveys (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, researcher TEXT, persona TEXT, timestamp TEXT, dialogue TEXT, raw_logs TEXT, answers TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS configs (researcher TEXT PRIMARY KEY, session_cfg TEXT, model_cfg TEXT, int_cfg TEXT, persona_cfg TEXT, secrets_cfg TEXT)')
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# USERS
# ──────────────────────────────────────────────
def get_user(username):
    """Returns dict with password, role, linked_researcher or None."""
    if DB_BACKEND == "firestore":
        doc = _get_firestore().collection("users").document(username).get()
        return doc.to_dict() if doc.exists else None
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password, role, linked_researcher FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"password": row[0], "role": row[1], "linked_researcher": row[2]}
    return None


def user_exists(username):
    return get_user(username) is not None


def create_user(username, hashed_password, role, linked_researcher):
    if DB_BACKEND == "firestore":
        _get_firestore().collection("users").document(username).set({
            "password": hashed_password,
            "role": role,
            "linked_researcher": linked_researcher
        })
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password, role, linked_researcher) VALUES (?, ?, ?, ?)",
              (username, hashed_password, role, linked_researcher))
    conn.commit()
    conn.close()


def get_researchers():
    """Returns list of researcher usernames."""
    if DB_BACKEND == "firestore":
        docs = _get_firestore().collection("users").where(filter=FieldFilter("role", "==", "researcher")).stream()
        return [doc.id for doc in docs]
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE role='researcher'")
    researchers = [r[0] for r in c.fetchall()]
    conn.close()
    return researchers


# ──────────────────────────────────────────────
# CONFIGS
# ──────────────────────────────────────────────
def get_configs(researcher):
    """Returns dict with session_cfg, model_cfg, int_cfg, persona_cfg, secrets_cfg (all as parsed objects) or None."""
    if DB_BACKEND == "firestore":
        doc = _get_firestore().collection("configs").document(researcher).get()
        return doc.to_dict() if doc.exists else None
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT session_cfg, model_cfg, int_cfg, persona_cfg, secrets_cfg FROM configs WHERE researcher=?",
              (researcher,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    result = {}
    keys = ["session_cfg", "model_cfg", "int_cfg", "persona_cfg", "secrets_cfg"]
    for i, key in enumerate(keys):
        try:
            result[key] = json.loads(row[i]) if row[i] else None
        except (json.JSONDecodeError, TypeError):
            result[key] = None
    return result


def upsert_configs(researcher, session_cfg=None, model_cfg=None, int_cfg=None, persona_cfg=None, secrets_cfg=None):
    """Insert or update config. Only non-None fields are updated."""
    fields = {
        "session_cfg": session_cfg,
        "model_cfg": model_cfg,
        "int_cfg": int_cfg,
        "persona_cfg": persona_cfg,
        "secrets_cfg": secrets_cfg
    }
    non_null = {k: v for k, v in fields.items() if v is not None}

    if DB_BACKEND == "firestore":
        doc_ref = _get_firestore().collection("configs").document(researcher)
        doc = doc_ref.get()
        if doc.exists:
            doc_ref.update(non_null)
        else:
            doc_ref.set(fields)
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT researcher FROM configs WHERE researcher=?", (researcher,))
    exists = c.fetchone()

    if exists:
        if non_null:
            updates = []
            values = []
            for col, val in non_null.items():
                updates.append(f"{col}=?")
                values.append(json.dumps(val))
            values.append(researcher)
            c.execute(f"UPDATE configs SET {', '.join(updates)} WHERE researcher=?", values)
    else:
        c.execute(
            "INSERT INTO configs (researcher, session_cfg, model_cfg, int_cfg, persona_cfg, secrets_cfg) VALUES (?, ?, ?, ?, ?, ?)",
            (researcher, json.dumps(session_cfg), json.dumps(model_cfg), json.dumps(int_cfg),
             json.dumps(persona_cfg), json.dumps(secrets_cfg)))
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# PROGRESS
# ──────────────────────────────────────────────
def get_completed_personas(username):
    """Returns list of completed persona names."""
    if DB_BACKEND == "firestore":
        docs = _get_firestore().collection("progress").where(filter=FieldFilter("username", "==", username)).stream()
        return [doc.to_dict().get("persona") for doc in docs]
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT persona FROM progress WHERE username=?", (username,))
    completed = [r[0] for r in c.fetchall()]
    conn.close()
    return completed


def mark_complete(username, persona_name):
    """Mark a persona as completed for a user. Idempotent."""
    if DB_BACKEND == "firestore":
        doc_id = f"{username}_{persona_name}"
        _get_firestore().collection("progress").document(doc_id).set({
            "username": username,
            "persona": persona_name
        })
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO progress (username, persona) VALUES (?, ?)", (username, persona_name))
    conn.commit()
    conn.close()


# ──────────────────────────────────────────────
# SURVEYS
# ──────────────────────────────────────────────
def save_survey(username, researcher, persona_name, timestamp, dialogue, raw_logs, answers):
    """Save survey results."""
    if DB_BACKEND == "firestore":
        _get_firestore().collection("surveys").add({
            "username": username,
            "researcher": researcher,
            "persona": persona_name,
            "timestamp": timestamp,
            "dialogue": dialogue,
            "raw_logs": raw_logs,
            "answers": answers
        })
        return
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO surveys (username, researcher, persona, timestamp, dialogue, raw_logs, answers) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, researcher, persona_name, timestamp,
         json.dumps(dialogue), json.dumps(raw_logs), json.dumps(answers)))
    conn.commit()
    conn.close()


def get_surveys(researcher, include_full=False):
    """Get survey results for a researcher. If include_full, includes dialogue and raw_logs."""
    if DB_BACKEND == "firestore":
        try:
            docs = _get_firestore().collection("surveys").where(filter=FieldFilter("researcher", "==", researcher)).stream()
            results = []
            for doc in docs:
                d = doc.to_dict()
                if include_full:
                    results.append(d)
                else:
                    results.append({
                        "id": doc.id,
                        "username": d.get("username"),
                        "persona": d.get("persona"),
                        "timestamp": d.get("timestamp"),
                        "answers": d.get("answers")
                    })
            # Sort by timestamp descending (avoids needing a Firestore composite index)
            results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
            return results
        except Exception as e:
            print(f"[DB] Error fetching surveys for researcher '{researcher}': {e}")
            return []

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if include_full:
        c.execute(
            "SELECT username, persona, timestamp, dialogue, raw_logs, answers FROM surveys WHERE researcher=? ORDER BY id DESC",
            (researcher,))
        results = [{"username": r[0], "persona": r[1], "timestamp": r[2], "dialogue": json.loads(r[3]),
                     "raw_logs": json.loads(r[4]), "answers": json.loads(r[5])} for r in c.fetchall()]
    else:
        c.execute("SELECT id, username, persona, timestamp, answers FROM surveys WHERE researcher=? ORDER BY id DESC",
                  (researcher,))
        results = [{"id": r[0], "username": r[1], "persona": r[2], "timestamp": r[3], "answers": json.loads(r[4])}
                   for r in c.fetchall()]
    conn.close()
    return results


# ──────────────────────────────────────────────
# SECRETS (Google Cloud Secret Manager)
# ──────────────────────────────────────────────
_secrets_client = None


def _get_secrets_client():
    global _secrets_client
    if _secrets_client is None:
        from google.cloud import secretmanager
        _secrets_client = secretmanager.SecretManagerServiceClient()
    return _secrets_client


def store_secret(researcher, secrets_dict):
    """Store researcher API keys in Secret Manager (production) or in configs table (dev)."""
    if DB_BACKEND != "firestore":
        # In dev mode, store in configs table as before
        upsert_configs(researcher, secrets_cfg=secrets_dict)
        return

    client = _get_secrets_client()
    project_id = os.environ.get("GCP_PROJECT_ID")
    secret_id = f"researcher-keys-{researcher}"
    parent = f"projects/{project_id}"
    secret_name = f"{parent}/secrets/{secret_id}"

    payload = json.dumps(secrets_dict).encode("UTF-8")

    try:
        client.get_secret(request={"name": secret_name})
    except Exception:
        client.create_secret(request={
            "parent": parent,
            "secret_id": secret_id,
            "secret": {"replication": {"automatic": {}}}
        })

    client.add_secret_version(request={
        "parent": secret_name,
        "payload": {"data": payload}
    })


def get_secret(researcher):
    """Retrieve researcher API keys from Secret Manager (production) or configs table (dev)."""
    if DB_BACKEND != "firestore":
        cfg = get_configs(researcher)
        return cfg.get("secrets_cfg", {}) if cfg else {}

    client = _get_secrets_client()
    project_id = os.environ.get("GCP_PROJECT_ID")
    secret_id = f"researcher-keys-{researcher}"
    secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

    try:
        response = client.access_secret_version(request={"name": secret_name})
        return json.loads(response.payload.data.decode("UTF-8"))
    except Exception:
        return {}
