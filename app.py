# --- CRITICAL FIX: MUST BE AT THE VERY TOP ---
from gevent import monkey

monkey.patch_all()
# ---------------------------------------------

import os
import sys
import json
import sqlite3
import hashlib
import re
import io
import copy
import base64
import random
import wave
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from openai import OpenAI
from elevenlabs.client import ElevenLabs

try:
    from google import genai
    from google.genai import types

    has_gemini = True
except ImportError:
    genai = None
    types = None
    has_gemini = False

app = Flask(__name__)
app.secret_key = "research_demo_super_secret_key"

# Updated to use gevent for high-speed concurrent processing
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

# --- DATABASE SETUP ---
DB_NAME = "research_demo.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT, linked_researcher TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS progress (username TEXT, persona TEXT, PRIMARY KEY(username, persona))''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS surveys (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, researcher TEXT, persona TEXT, timestamp TEXT, dialogue TEXT, raw_logs TEXT, answers TEXT)''')
    c.execute(
        '''CREATE TABLE IF NOT EXISTS configs (researcher TEXT PRIMARY KEY, session_cfg TEXT, model_cfg TEXT, int_cfg TEXT, persona_cfg TEXT, secrets_cfg TEXT)''')
    conn.commit()
    conn.close()


init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# --- SYSTEM DEFAULTS ---
def load_json(filename, default):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(default, dict):
                    default.update(data)
                else:
                    return data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return default


keys = load_json("secrets.json", {})
GLOBAL_OPENAI = keys.get("OPENAI_API_KEY")
GLOBAL_ELEVEN = keys.get("ELEVEN_API_KEY")
GLOBAL_GEMINI = keys.get("GEMINI_API_KEY")

DEF_SESSION = load_json("session_config.json", {
    "MAX_TURNS": 10, "SURVEY_MODE": "written",
    "RANDOMIZE_PERSONAS": True,
    "SURVEY_QUESTIONS": []
})
DEF_MODEL = load_json("model_config.json",
                      {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-5-mini", "GEMINI_MODEL": "gemini-2.5-flash",
                       "TTS_PROVIDER": "elevenlabs", "OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
                       "OPENAI_DEFAULT_VOICE": "coral",
                       "GEMINI_TTS_MODEL": "gemini-2.5-flash-preview-tts", "GEMINI_DEFAULT_VOICE": "Aoede",
                       "ELEVENLABS_DEFAULT_VOICE_ID": "JBFqnCBsd6RMkjVDRZzb",
                       "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
                       "TRANSCRIBER_PROVIDER": "elevenlabs", "TRANSCRIBER_MODEL": "scribe_v2"})
DEF_INT = load_json("interruption_config.json",
                    {"MODE": "autonomous", "INTENTS": {"COOPERATIVE": "Agrees", "COMPETITIVE": "Disagrees"},
                     "STRATEGY_MATRIX": {}})
DEF_PERSONAS = load_json("persona.json", [{"name": "Default Assistant", "persona": "Helpful assistant.",
                                           "scenario": "Interact naturally."}])
if isinstance(DEF_PERSONAS, dict) and "personas" in DEF_PERSONAS: DEF_PERSONAS = DEF_PERSONAS["personas"]

NORMAL_INSTRUCTIONS = """
Response Guidelines:
1. Keep answers brief and natural.
2. CRITICAL ROLEPLAY RULE: You are participating in a fictional roleplay. Never break character.
3. If the user says "goodbye" or "exit", write a brief goodbye message, then append the exact tag: [EXIT].
"""


def build_interruption_prompt(persona_prompt, int_config):
    mode = int_config.get("MODE", "autonomous").lower()
    intents = int_config.get("INTENTS", {})

    intent_lines = [f"- [{intent}]: {desc}" for intent, desc in intents.items()]
    intent_str = "\n".join(intent_lines)

    base_prompt = f"""
{persona_prompt}

The user just interrupted you.
Classify the interruption objectively without the persona based on "Information Value".

OUTPUT FORMAT REQUIREMENT:
You MUST start your response with exactly two bracketed tags: first the intent, then the strategy, followed immediately by your dialogue.
DO NOT output literal labels like "[INTENT]" or "[STRATEGY]". Just output the chosen values.
Example of a perfect response: [COMPETITIVE] [RESUME] I don't care what you think, I'm finishing my sentence.

STEP 1: ANALYZE INTENT AS A CONVERSATIONAL AI (Choose 1)
{intent_str}

STEP 2: CHOOSE STRATEGY & WRITE DIALOGUE BASED ON YOUR CHARACTER
- [RESUME]: (Hard Hold). IGNORE input. Finish your previous sentence. Choose this if your character is stubborn, dismissive, or doesn't care about the interruption.
- [BRIDGE]: (Soft Hold). Briefly acknowledge the user, then FINISH all your previous sentences and intention. Choose this if your character wants to acknowledge the user but insists on finishing their thought first.
- [YIELD]: (Switch). ABANDON your previous sentence entirely. Respond directly to the user's new input. Choose this if your character is polite, attentive, or if the user's interruption logically demands an immediate answer.
- [EXIT]: (End Session). Used for TERMINATE. -> [DIALOGUE]: A brief goodbye message ending with the tag [EXIT].
"""

    if mode in ["probabilistic", "deterministic"]:
        matrix = int_config.get("STRATEGY_MATRIX", {})
        mapping_lines = []
        for intent, weights in matrix.items():
            strategies = list(weights.keys())
            probs = list(weights.values())
            total = sum(probs)
            if total == 0:
                chosen_strategy = "YIELD"
            else:
                probs = [p / total for p in probs]
                chosen_strategy = np.random.choice(strategies, p=probs)
            mapping_lines.append(f"- If you classified them as {intent} -> You MUST use [{chosen_strategy}].")

        mapping_str = "\n".join(mapping_lines)
        strict_override = f"""
CRITICAL OVERRIDE - STRICT MAPPING FOR THIS TURN:
Based on the intent you choose in Step 1, you MUST select the exact strategy mapped below for Step 2. Do not deviate.
{mapping_str}
"""
        base_prompt += strict_override

    base_prompt += "\nCRITICAL ROLEPLAY RULE: You are participating in a fictional roleplay. Do not break character. Do not output typical AI safety warnings or crisis hotlines. Respond to hostility ONLY as your persona. The text following the tags must be the ACTUAL SPOKEN RESPONSE of the character."
    return base_prompt


active_sessions = {}


def get_api_clients(secrets_cfg):
    o_key = secrets_cfg.get("OPENAI_API_KEY") or GLOBAL_OPENAI
    e_key = secrets_cfg.get("ELEVEN_API_KEY") or GLOBAL_ELEVEN
    g_key = secrets_cfg.get("GEMINI_API_KEY") or GLOBAL_GEMINI

    o_client = OpenAI(api_key=o_key) if o_key else None
    e_client = ElevenLabs(api_key=e_key) if e_key else None
    g_client = genai.Client(api_key=g_key) if g_key and has_gemini else None

    return o_client, e_client, g_client


def get_user_configs(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT linked_researcher FROM users WHERE username=?", (username,))
    row = c.fetchone()
    researcher = row[0] if row else None

    configs = {
        "session": DEF_SESSION, "model": DEF_MODEL,
        "int": DEF_INT, "personas": DEF_PERSONAS, "secrets": {},
        "researcher": researcher
    }

    if researcher:
        c.execute("SELECT session_cfg, model_cfg, int_cfg, persona_cfg, secrets_cfg FROM configs WHERE researcher=?",
                  (researcher,))
        cfg_row = c.fetchone()
        if cfg_row:
            try:
                if cfg_row[0]: configs["session"] = json.loads(cfg_row[0])
                if cfg_row[1]: configs["model"] = json.loads(cfg_row[1])
                if cfg_row[2]: configs["int"] = json.loads(cfg_row[2])
                if cfg_row[3]:
                    p_data = json.loads(cfg_row[3])
                    configs["personas"] = p_data.get("personas", p_data) if isinstance(p_data, dict) else p_data
                if cfg_row[4]: configs["secrets"] = json.loads(cfg_row[4])
            except Exception as e:
                print(f"Failed to parse custom config for {researcher}, using defaults. Error: {e}")
    conn.close()
    return configs


# --- ROUTING & API ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/researchers', methods=['GET'])
def get_researchers():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE role='researcher'")
    researchers = [r[0] for r in c.fetchall()]
    conn.close()
    return jsonify({"researchers": researchers})


@app.route('/api/auth', methods=['POST'])
def auth():
    data = request.json
    action, username, password = data.get('action'), data.get('username'), data.get('password')
    role, linked_researcher = data.get('role', 'tester'), data.get('linked_researcher', None)

    if not username or not password: return jsonify({"success": False, "msg": "Fields cannot be empty."})

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    if action == 'signup':
        c.execute("SELECT username FROM users WHERE username=?", (username,))
        if c.fetchone(): return jsonify({"success": False, "msg": "Username exists."})
        c.execute("INSERT INTO users (username, password, role, linked_researcher) VALUES (?, ?, ?, ?)",
                  (username, hash_password(password), role, linked_researcher))
        conn.commit()
        res = {"success": True, "msg": "Signup successful. Please log in."}
    else:
        c.execute("SELECT password, role FROM users WHERE username=?", (username,))
        row = c.fetchone()
        if row and row[0] == hash_password(password):
            res = {"success": True, "msg": "Login successful.", "role": row[1]}
        else:
            res = {"success": False, "msg": "Invalid credentials."}
    conn.close()
    return jsonify(res)


@app.route('/api/configs', methods=['POST'])
def upload_configs():
    data = request.json
    researcher = data.get('researcher')
    if not researcher: return jsonify({"success": False})

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO configs (researcher, session_cfg, model_cfg, int_cfg, persona_cfg, secrets_cfg) VALUES (?, ?, ?, ?, ?, ?)",
        (researcher, json.dumps(data.get('session')), json.dumps(data.get('model')), json.dumps(data.get('int')),
         json.dumps(data.get('persona')), json.dumps(data.get('secrets'))))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route('/api/results', methods=['GET'])
def get_results():
    researcher = request.args.get('researcher')
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, persona, timestamp, answers FROM surveys WHERE researcher=? ORDER BY id DESC",
              (researcher,))
    results = [{"id": r[0], "username": r[1], "persona": r[2], "timestamp": r[3], "answers": json.loads(r[4])} for r in
               c.fetchall()]
    conn.close()
    return jsonify(results)


@app.route('/api/export', methods=['GET'])
def export_results():
    researcher = request.args.get('researcher')
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT username, persona, timestamp, dialogue, raw_logs, answers FROM surveys WHERE researcher=? ORDER BY id DESC",
        (researcher,))
    results = [{"username": r[0], "persona": r[1], "timestamp": r[2], "dialogue": json.loads(r[3]),
                "raw_logs": json.loads(r[4]), "answers": json.loads(r[5])} for r in c.fetchall()]
    conn.close()
    return Response(json.dumps(results, indent=4), mimetype="application/json",
                    headers={"Content-disposition": f"attachment; filename=research_results_{researcher}.json"})


# --- SOCKET IO HANDLERS ---
@socketio.on('connect')
def handle_connect():
    active_sessions[request.sid] = {
        "username": None, "persona_idx": 0, "history": [], "raw_logs": [],
        "turn": 0, "last_interruption": "", "bot_text": "", "remaining_text": "",
        "is_speaking": False, "is_processing": False,
        "configs": None
    }


@socketio.on('start_experiment')
def start_experiment(data):
    sid = request.sid
    username = data.get('username')

    session_configs = get_user_configs(username)
    active_sessions[sid]["configs"] = session_configs
    active_sessions[sid]["username"] = username

    personas_list = session_configs["personas"]

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT persona FROM progress WHERE username=?", (username,))
    completed = [r[0] for r in c.fetchall()]
    conn.close()

    uncompleted_indices = [i for i, p in enumerate(personas_list) if p.get("name", "") not in completed]

    if not uncompleted_indices:
        emit('set_persona',
             {"name": "All Sessions Complete", "scenario": "Thank you for participating. You may close this window."})
        emit('status', {"msg": "All personas completed!", "color": "#00FF00"})
        return

    if session_configs["session"].get("RANDOMIZE_PERSONAS", True):
        idx = random.choice(uncompleted_indices)
    else:
        idx = uncompleted_indices[0]

    active_sessions[sid]["persona_idx"] = idx
    active_sessions[sid]["history"] = []
    active_sessions[sid]["raw_logs"] = [f"=== STARTING SESSION: {personas_list[idx].get('name')} ==="]
    active_sessions[sid]["turn"] = 0
    active_sessions[sid]["last_interruption"] = ""
    active_sessions[sid]["remaining_text"] = ""
    active_sessions[sid]["is_processing"] = True

    persona = personas_list[idx]
    emit('set_persona', {"name": persona.get('name'), "scenario": persona.get('scenario', 'Interact naturally.')})
    emit('status', {"msg": f"Starting Session: {persona.get('name')}", "color": "magenta"})

    prompt = f"{persona.get('persona', '')}\nGive a very short, 1-sentence opening greeting."
    socketio.start_background_task(run_bot_turn, sid, [{"role": "developer", "content": prompt}], persona,
                                   session_configs, True)


@socketio.on('user_audio_blob')
def handle_user_audio(audio_data):
    sid = request.sid
    session_state = active_sessions.get(sid)

    if session_state and session_state.get('is_processing'):
        return

    if session_state:
        session_state['is_processing'] = True

    socketio.start_background_task(process_user_audio, sid, audio_data)


def process_user_audio(sid, audio_data):
    session_state = active_sessions.get(sid)
    if not session_state or not session_state["username"]:
        if session_state: session_state['is_processing'] = False
        return

    cfg = session_state["configs"]
    persona = cfg["personas"][session_state["persona_idx"]]
    session_limit = cfg["session"].get("MAX_TURNS", 5)

    o_client, e_client, _ = get_api_clients(cfg.get("secrets", {}))

    audio_buffer = io.BytesIO(audio_data)
    audio_buffer.name = "audio.webm"

    socketio.emit('status', {"msg": "Transcribing...", "color": "yellow"}, to=sid)
    user_text = ""
    transcription_success = False

    transcriber_provider = cfg["model"].get("TRANSCRIBER_PROVIDER", "elevenlabs").lower()
    transcriber_model = cfg["model"].get("TRANSCRIBER_MODEL", "scribe_v2")

    def try_elevenlabs():
        if e_client:
            t_model = transcriber_model if transcriber_provider == "elevenlabs" else "scribe_v2"
            audio_buffer.seek(0)
            res = e_client.speech_to_text.convert(file=audio_buffer, model_id=t_model, tag_audio_events=False)
            return res.text
        return None

    def try_openai():
        if o_client:
            t_model = transcriber_model if transcriber_provider == "openai" else "gpt-4o-mini-transcribe"
            audio_buffer.seek(0)
            res = o_client.audio.transcriptions.create(model=t_model, file=audio_buffer)
            return res.text
        return None

    if transcriber_provider == "openai":
        try:
            user_text = try_openai()
            transcription_success = user_text is not None
        except Exception as e:
            if not transcription_success:
                try:
                    user_text = try_elevenlabs(); transcription_success = user_text is not None
                except Exception as fallback_e:
                    pass
    else:
        try:
            user_text = try_elevenlabs()
            transcription_success = user_text is not None
        except Exception as e:
            if not transcription_success:
                try:
                    user_text = try_openai(); transcription_success = user_text is not None
                except Exception as fallback_e:
                    pass

    if not transcription_success and not e_client and not o_client:
        socketio.emit('status', {"msg": "No API Keys provided for transcription.", "color": "red"}, to=sid)

    if not user_text or not user_text.strip():
        if session_state["last_interruption"]:
            user_text = "*interrupted you but mumbled or trailed off*"
        else:
            session_state['is_processing'] = False
            socketio.emit('status', {"msg": "Listening...", "color": "#666"}, to=sid)
            return

    socketio.emit('chat', {"speaker": "User", "text": user_text}, to=sid)
    session_state["history"].append({"role": "user", "content": user_text})
    session_state["raw_logs"].append(f"User Spoke: {user_text}")
    session_state["turn"] += 1

    messages = []

    current_interruption_cfg = copy.deepcopy(cfg["int"])
    if "interruption_config" in persona:
        p_config = persona["interruption_config"]
        if "MODE" in p_config: current_interruption_cfg["MODE"] = p_config["MODE"]
        if "INTENTS" in p_config: current_interruption_cfg["INTENTS"].update(p_config["INTENTS"])
        if "STRATEGY_MATRIX" in p_config:
            for intent, matrix in p_config["STRATEGY_MATRIX"].items():
                if intent in current_interruption_cfg["STRATEGY_MATRIX"]:
                    current_interruption_cfg["STRATEGY_MATRIX"][intent].update(matrix)
                else:
                    current_interruption_cfg["STRATEGY_MATRIX"][intent] = matrix

    if session_state["last_interruption"]:
        dynamic_interruption_prompt = build_interruption_prompt(persona.get('persona'), current_interruption_cfg)
        messages.append({"role": "developer",
                         "content": f"{dynamic_interruption_prompt}\n\n--- IMMEDIATE CONTEXT ---\n{session_state['last_interruption']}"})
        session_state["last_interruption"] = ""
    else:
        messages.append({"role": "developer", "content": f"{persona.get('persona')}\n{NORMAL_INSTRUCTIONS}"})

    messages.extend(session_state["history"])

    run_bot_turn(sid, messages, persona, cfg, False)


def run_bot_turn(sid, messages, persona, cfg, is_greeting):
    session_state = active_sessions.get(sid)
    session_limit = cfg["session"].get("MAX_TURNS", 5)

    socketio.emit('status', {"msg": "LLM Thinking...", "color": "yellow"}, to=sid)
    bot_response = process_llm(messages, persona, cfg)

    found_tags = [t.upper() for t in re.findall(r'\[([a-zA-Z_]+)\]', bot_response)]
    valid_intents = ["BACKCHANNEL", "COOPERATIVE", "COMPETITIVE", "TOPIC_CHANGE", "TERMINATE"]
    valid_strategies = ["RESUME", "BRIDGE", "YIELD", "EXIT", "OVERRULE"]

    intent = next((t for t in found_tags if t in valid_intents), "UNKNOWN")
    strategy = next((t for t in found_tags if t in valid_strategies), "YIELD")

    clean_response = re.sub(r'\[.*?\]', '', bot_response).strip()
    should_exit = "EXIT" in found_tags or strategy == "EXIT"

    if strategy == "RESUME" and session_state.get("remaining_text") and not is_greeting:
        clean_response = "..." + session_state["remaining_text"]

    session_state["raw_logs"].append(f"LLM Raw Output: {bot_response}")
    session_state["raw_logs"].append(f"Bot Decision -> Intent: [{intent}] | Strategy: [{strategy}]")
    session_state["raw_logs"].append(f"Bot Spoke: {clean_response}")
    session_state["history"].append({"role": "assistant", "content": bot_response})

    socketio.emit('chat', {"speaker": "Bot", "text": clean_response}, to=sid)

    if should_exit or session_state["turn"] >= session_limit:
        send_audio_buffer(sid, clean_response, persona, cfg, is_final=True)
    else:
        send_audio_buffer(sid, clean_response, persona, cfg, is_final=False)


@socketio.on('interrupt_signal')
def handle_interrupt(data):
    sid = request.sid
    session_state = active_sessions.get(sid)
    if session_state:
        session_state['is_processing'] = False

        if session_state.get("is_speaking"):
            session_state["is_speaking"] = False
            percent_played = data.get("percent", 0.5)
            if percent_played > 1.0: percent_played = 0.99

            full_text = session_state["bot_text"]
            char_idx = int(len(full_text) * percent_played)
            cutoff = full_text[:char_idx]
            remain = full_text[char_idx:]

            interruption_msg = f"SYSTEM NOTE: You were interrupted. You were saying: '{cutoff}'. You intended to say: '{remain}'."
            session_state["last_interruption"] = interruption_msg
            session_state["remaining_text"] = remain
            session_state["raw_logs"].append(f"!!! INTERRUPTION !!! Cutoff: '{cutoff}' | Remaining: '{remain}'")

            emit('status', {"msg": f"Interrupted at: {cutoff}...", "color": "red"})


@socketio.on('submit_survey')
def handle_survey(data):
    sid = request.sid
    session_state = active_sessions.get(sid)
    username = session_state["username"]
    cfg = session_state["configs"]
    researcher = cfg["researcher"]
    persona_name = cfg["personas"][session_state["persona_idx"]].get("name")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO surveys (username, researcher, persona, timestamp, dialogue, raw_logs, answers) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, researcher, persona_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         json.dumps(session_state["history"]), json.dumps(session_state["raw_logs"]), json.dumps(data)))
    c.execute("INSERT INTO progress (username, persona) VALUES (?, ?)", (username, persona_name))
    conn.commit()
    conn.close()

    emit('status', {"msg": "Survey saved. Loading next persona...", "color": "#00FF00"})
    start_experiment({"username": username})


# --- HELPER FUNCTIONS ---
def process_llm(messages, persona, cfg):
    model_cfg = cfg["model"]
    provider = persona.get("llm_provider", model_cfg.get("LLM_PROVIDER", "openai")).lower()

    o_client, _, g_client = get_api_clients(cfg.get("secrets", {}))

    try:
        if provider == "openai" and o_client:
            res = o_client.chat.completions.create(model=model_cfg.get("OPENAI_MODEL", "gpt-5-mini"), messages=messages)
            return res.choices[0].message.content
        elif provider == "gemini" and g_client:
            sys_msg = "\n".join([m["content"] for m in messages if m["role"] == "developer"])
            gem_msgs = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in
                        messages if m["role"] != "developer"]
            if not gem_msgs:
                gem_msgs = [{"role": "user", "parts": [{"text": "Hello."}]}]
            elif gem_msgs[0]["role"] == "model":
                gem_msgs.insert(0, {"role": "user", "parts": [{"text": "Hello."}]})
            conf = types.GenerateContentConfig(system_instruction=sys_msg)
            res = g_client.models.generate_content(model=model_cfg.get("GEMINI_MODEL", "gemini-2.5-flash"),
                                                   contents=gem_msgs, config=conf)
            return res.text
        else:
            return "Configuration Error: Valid API key not found for the requested LLM provider."
    except Exception as e:
        print(f"LLM Error: {e}")
        return "I encountered a cognitive error."


def send_audio_buffer(sid, text, persona, cfg, is_final=False):
    model_cfg = cfg["model"]
    provider = persona.get("tts_provider", model_cfg.get("TTS_PROVIDER", "elevenlabs")).lower()

    o_client, e_client, g_client = get_api_clients(cfg.get("secrets", {}))
    socketio.emit('status', {"msg": "Generating Speech...", "color": "yellow"}, to=sid)

    audio_data = None
    try:
        if provider == "openai" and o_client:
            res = o_client.audio.speech.create(model=model_cfg.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                                               voice=persona.get("voice_id",
                                                                 model_cfg.get("OPENAI_DEFAULT_VOICE", "coral")),
                                               input=text, response_format="mp3")
            audio_data = res.content
        elif provider == "elevenlabs" and e_client:
            stream = e_client.text_to_speech.convert(text=text,
                                                     voice_id=persona.get("voice_id",
                                                                          model_cfg.get("ELEVENLABS_DEFAULT_VOICE_ID",
                                                                                        "JBFqnCBsd6RMkjVDRZzb")),
                                                     model_id=model_cfg.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
                                                     output_format="mp3_44100_128")
            audio_data = b"".join([chunk for chunk in stream])
        elif provider == "gemini" and g_client:
            conf = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=persona.get("voice_id", model_cfg.get("GEMINI_DEFAULT_VOICE", "Aoede")))))
            )
            res = g_client.models.generate_content(
                model=model_cfg.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"), contents=text, config=conf)

            for part in res.candidates[0].content.parts:
                if getattr(part, "inline_data", None):
                    raw_pcm = part.inline_data.data
                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(raw_pcm)
                    audio_data = buffer.getvalue()
                    break
    except Exception as e:
        print(f"TTS Error: {e}")

    if audio_data:
        active_sessions[sid]["bot_text"] = text
        active_sessions[sid]["is_speaking"] = True

        payload = {
            "audio_b64": base64.b64encode(audio_data).decode('utf-8'),
            "duration_est": len(text) * 0.05,
            "is_final": is_final
        }
        if is_final:
            payload["survey"] = cfg["session"].get("SURVEY_QUESTIONS", [])

        socketio.emit('play_audio_buffer', payload, to=sid)
        socketio.emit('status', {"msg": "Speaking...", "color": "cyan"}, to=sid)

    active_sessions[sid]['is_processing'] = False


if __name__ == '__main__':
    try:
        print("--- Initializing Server on Localhost:5050 ---")
        # use_reloader=False prevents Werkzeug from forking the process, fixing the gevent crash
        socketio.run(app, debug=True, use_reloader=False, host="127.0.0.1", port=5050, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server Stopped.")