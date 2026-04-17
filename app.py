import os
import sys
import json
import hashlib
import re
import io
import copy
import base64
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

import db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "research_demo_super_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*", ping_interval=20, ping_timeout=60)

db.init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# --- SYSTEM DEFAULTS (Used if researcher doesn't upload their own) ---
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


# Global fallback keys (optional for public deployments)
keys = load_json("secrets.json", {})
GLOBAL_OPENAI = keys.get("OPENAI_API_KEY")
GLOBAL_ELEVEN = keys.get("ELEVEN_API_KEY")
GLOBAL_GEMINI = keys.get("GEMINI_API_KEY")

DEF_SESSION = load_json("session_config.json", {
    "MAX_TURNS": 5, "SURVEY_MODE": "written", "SURVEY_TYPE": "individual", "SET_SIZE": 1, "SET_LABELS": [],
    "SURVEY_QUESTIONS": [
        {"question": "How natural was the bot's flow (1-5)?", "expected_type": "number",
         "allowed_options": ["1", "2", "3", "4", "5"]},
        {"question": "Did it handle interruptions well? (yes/no)", "expected_type": "options",
         "allowed_options": ["yes", "no"]}
    ]
})
DEF_MODEL = load_json("model_config.json",
                      {"LLM_PROVIDER": "gemini", "OPENAI_MODEL": "gpt-4o-mini", "GEMINI_MODEL": "gemini-2.5-flash",
                       "TTS_PROVIDER": "openai", "OPENAI_TTS_MODEL": "gpt-4o-mini-tts", "OPENAI_DEFAULT_VOICE": "coral",
                       "GEMINI_TTS_MODEL": "gemini-2.5-flash-preview-tts", "GEMINI_DEFAULT_VOICE": "Aoede",
                       "ELEVENLABS_DEFAULT_VOICE_ID": "JBFqnCBsd6RMkjVDRZzb",
                       "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5"})
DEF_INT = load_json("interruption_config.json",
                    {"MODE": "autonomous", "INTENTS": {"COOPERATIVE": "Agrees", "COMPETITIVE": "Disagrees"},
                     "STRATEGY_MATRIX": {}})
DEF_PERSONAS = load_json("persona.json", [{"name": "Default Assistant", "persona": "Helpful assistant."}])
if isinstance(DEF_PERSONAS, dict) and "personas" in DEF_PERSONAS: DEF_PERSONAS = DEF_PERSONAS["personas"]

NORMAL_INSTRUCTIONS = """
Response Guidelines:
1. Keep answers brief and natural.
2. CRITICAL ROLEPLAY RULE: You are participating in a fictional roleplay. Never break character.
3. Never end the conversation on your own. Keep the dialogue going until the user explicitly wants to leave.
4. If the user says "goodbye", "exit", "I'm done", or clearly wants to end the conversation, write a brief in-character farewell, then append the exact tag: [EXIT].
"""


def build_interruption_prompt(persona_prompt, int_config):
    mode = int_config.get("MODE", "autonomous").lower()
    intents = int_config.get("INTENTS", {})

    intent_lines = []
    for intent, desc in intents.items():
        intent_lines.append(f"- [{intent}]: {desc}")
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
- [OVERRIDE]: (Forced Hold). Like RESUME but STRONGER. Finish your previous sentence AND add new content. The user will NOT be able to interrupt your response this time. Choose this only when your character would forcefully demand to be heard without any possibility of being cut off.
- [EXIT]: (End Session). Used ONLY for TERMINATE intent. Write a brief in-character farewell, then append the tag [EXIT].
- NOTE: Do NOT end the conversation yourself unless the user clearly wants to leave (TERMINATE).
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
user_sessions = {}  # Backup keyed by username — survives WebSocket reconnect


# --- HELPER: DYNAMIC API CLIENT ROUTING ---
def get_api_clients(secrets_cfg):
    """Instantiates the AI clients dynamically using the researcher's uploaded keys."""
    o_key = secrets_cfg.get("OPENAI_API_KEY") or GLOBAL_OPENAI
    e_key = secrets_cfg.get("ELEVEN_API_KEY") or GLOBAL_ELEVEN
    g_key = secrets_cfg.get("GEMINI_API_KEY") or GLOBAL_GEMINI

    o_client = OpenAI(api_key=o_key) if o_key else None
    e_client = ElevenLabs(api_key=e_key) if e_key else None
    g_client = genai.Client(api_key=g_key) if g_key and has_gemini else None

    return o_client, e_client, g_client


# --- HELPER: GET RESEARCHER CONFIGS ---
def get_user_configs(username):
    user = db.get_user(username)
    researcher = user.get("linked_researcher") if user else None

    configs = {
        "session": copy.deepcopy(DEF_SESSION), "model": copy.deepcopy(DEF_MODEL),
        "int": copy.deepcopy(DEF_INT), "personas": copy.deepcopy(DEF_PERSONAS), "secrets": {},
        "researcher": researcher
    }

    if researcher:
        cfg_row = db.get_configs(researcher)
        if cfg_row:
            try:
                if cfg_row.get("session_cfg"): configs["session"] = cfg_row["session_cfg"]
                if cfg_row.get("model_cfg"): configs["model"] = cfg_row["model_cfg"]
                if cfg_row.get("int_cfg"): configs["int"] = cfg_row["int_cfg"]
                if cfg_row.get("persona_cfg"):
                    p_data = cfg_row["persona_cfg"]
                    configs["personas"] = p_data.get("personas", p_data) if isinstance(p_data, dict) else p_data
            except Exception as e:
                print(f"Failed to parse custom config for {researcher}, using defaults. Error: {e}")
        # Load secrets separately (from Secret Manager in production)
        configs["secrets"] = db.get_secret(researcher)
    return configs


# --- ROUTING & API ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/posting')
def posting():
    return render_template('posting.html')


@app.route('/api/researchers', methods=['GET'])
def get_researchers():
    return jsonify({"researchers": db.get_researchers()})


@app.route('/api/auth', methods=['POST'])
def auth():
    data = request.json
    action, username, password = data.get('action'), data.get('username'), data.get('password')
    role, linked_researcher = data.get('role', 'tester'), data.get('linked_researcher', None)

    if not username or not password: return jsonify({"success": False, "msg": "Fields cannot be empty."})

    if action == 'signup':
        if db.user_exists(username):
            return jsonify({"success": False, "msg": "Username exists."})
        db.create_user(username, hash_password(password), role, linked_researcher)
        res = {"success": True, "msg": "Signup successful. Please log in."}
    else:
        user = db.get_user(username)
        if user and user["password"] == hash_password(password):
            res = {"success": True, "msg": "Login successful.", "role": user["role"]}
        else:
            res = {"success": False, "msg": "Invalid credentials."}
    return jsonify(res)


@app.route('/api/configs', methods=['POST'])
def upload_configs():
    data = request.json
    researcher = data.get('researcher')
    if not researcher: return jsonify({"success": False})

    # Store secrets separately (Secret Manager in production)
    secrets = data.get('secrets')
    if secrets:
        db.store_secret(researcher, secrets)

    # Store non-secret configs
    db.upsert_configs(
        researcher,
        session_cfg=data.get('session'),
        model_cfg=data.get('model'),
        int_cfg=data.get('int'),
        persona_cfg=data.get('persona')
    )
    return jsonify({"success": True})


@app.route('/api/results', methods=['GET'])
def get_results():
    researcher = request.args.get('researcher')
    results = db.get_surveys(researcher, include_full=False)
    return jsonify(results)


@app.route('/api/export', methods=['GET'])
def export_results():
    researcher = request.args.get('researcher')
    results = db.get_surveys(researcher, include_full=True)
    return Response(
        json.dumps(results, indent=4),
        mimetype="application/json",
        headers={"Content-disposition": f"attachment; filename=research_results_{researcher}.json"}
    )


# --- SOCKET IO HANDLERS ---
@socketio.on('connect')
def handle_connect():
    active_sessions[request.sid] = {
        "username": None, "persona_idx": 0, "history": [], "raw_logs": [],
        "turn": 0, "last_interruption": "", "bot_text": "", "remaining_text": "", "is_speaking": False, "uninterruptible": False,
        "configs": None,
        # Comparative survey tracking
        "set_dialogues": [], "set_raw_logs": [], "set_personas": []
    }


@socketio.on('ping_keepalive')
def handle_ping():
    pass  # Just keeping the connection alive


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    session_state = active_sessions.pop(sid, None)
    if session_state and session_state.get("username"):
        username = session_state["username"]
        print(f"[{sid[:8]}] DISCONNECT: saving session for user '{username}' to user_sessions")
        user_sessions[username] = session_state


@socketio.on('rejoin_session')
def handle_rejoin(data):
    sid = request.sid
    username = data.get('username')
    if not username:
        emit('rejoin_result', {"success": False})
        return

    saved = user_sessions.pop(username, None)
    if saved and saved.get("configs"):
        print(f"[{sid[:8]}] REJOIN: restoring session for user '{username}' | in_survey={saved.get('in_survey')} | persona_idx={saved.get('persona_idx')} | set_personas={saved.get('set_personas')}")
        active_sessions[sid] = saved
        cfg = saved["configs"]
        survey_type = cfg["session"].get("SURVEY_TYPE", "individual")
        set_size = cfg["session"].get("SET_SIZE", 1)
        set_labels = cfg["session"].get("SET_LABELS", [])

        # If session was mid-survey, route to the appropriate screen
        if saved.get("in_survey"):
            set_personas = saved.get("set_personas", [])
            if survey_type == "comparative" and set_size > 1:
                if len(set_personas) >= set_size:
                    # Full set complete — need survey
                    print(f"[{sid[:8]}] REJOIN: session needs survey (set complete)")
                    emit('rejoin_result', {"success": True, "needs_survey": True})
                    _emit_survey_fallback(cfg, saved)
                    return
                else:
                    # Mid-set — show transition
                    current_pos = len(set_personas)
                    current_label = set_labels[current_pos - 1] if current_pos - 1 < len(set_labels) else f"Session {current_pos}"
                    next_label = set_labels[current_pos] if current_pos < len(set_labels) else f"Session {current_pos + 1}"
                    print(f"[{sid[:8]}] REJOIN: session needs transition ({current_label} -> {next_label})")
                    emit('rejoin_result', {"success": True})
                    emit('show_transition', {
                        "completed_label": current_label,
                        "next_label": next_label,
                        "current_pos": current_pos,
                        "set_size": set_size
                    })
                    return
            else:
                # Individual mode — need survey
                print(f"[{sid[:8]}] REJOIN: session needs survey (individual)")
                emit('rejoin_result', {"success": True, "needs_survey": True})
                _emit_survey_fallback(cfg, saved)
                return

        # Normal case — restore UI and resume listening
        persona = cfg["personas"][saved["persona_idx"]]
        if survey_type == "comparative" and set_size > 1:
            current_set_pos = len(saved.get("set_personas", []))
            label = set_labels[current_set_pos] if current_set_pos < len(set_labels) else f"Session {current_set_pos + 1}"
            emit('set_persona', {"name": label})
        else:
            emit('set_persona', {"name": persona.get('name')})

        # Replay chat history
        for msg in saved.get("history", []):
            speaker = "Bot" if msg["role"] == "assistant" else "User"
            emit('chat', {"speaker": speaker, "text": msg["content"]})

        emit('rejoin_result', {"success": True})
        emit('status', {"msg": "Session restored! Listening...", "color": "#00FF00"})
    else:
        print(f"[{sid[:8]}] REJOIN: no saved session for user '{username}'")
        emit('rejoin_result', {"success": False})


@socketio.on('start_experiment')
def start_experiment(data):
    sid = request.sid
    username = data.get('username')

    session_configs = get_user_configs(username)
    active_sessions[sid]["configs"] = session_configs
    active_sessions[sid]["username"] = username

    personas_list = session_configs["personas"]
    set_size = session_configs["session"].get("SET_SIZE", 1)

    completed = db.get_completed_personas(username)

    # Find the first set where not ALL personas are completed
    # Progress is only saved per-set (on survey submit), so partial sets restart from Style A
    idx = 0
    while idx < len(personas_list):
        set_start = idx
        set_end = min(idx + set_size, len(personas_list))
        set_names = [p.get("name", "") for p in personas_list[set_start:set_end]]
        if all(name in completed for name in set_names):
            idx = set_end  # Entire set is done, skip to next
        else:
            idx = set_start  # This set is incomplete, restart from its beginning
            break

    if idx >= len(personas_list):
        emit('set_persona', {"name": "All Sessions Complete"})
        emit('status', {"msg": "All personas completed!", "color": "#00FF00"})
        return

    active_sessions[sid]["persona_idx"] = idx
    active_sessions[sid]["history"] = []
    active_sessions[sid]["raw_logs"] = [f"=== STARTING SESSION: {personas_list[idx].get('name')} ==="]
    active_sessions[sid]["turn"] = 0
    active_sessions[sid]["last_interruption"] = ""
    active_sessions[sid]["remaining_text"] = ""
    active_sessions[sid]["in_survey"] = False

    # Always reset set tracking (we always start from the beginning of a set)
    active_sessions[sid]["set_dialogues"] = []
    active_sessions[sid]["set_raw_logs"] = []
    active_sessions[sid]["set_personas"] = []

    persona = personas_list[idx]

    survey_type = session_configs["session"].get("SURVEY_TYPE", "individual")
    set_labels = session_configs["session"].get("SET_LABELS", [])
    if survey_type == "comparative" and set_size > 1:
        current_set_pos = 0  # Always starting from beginning of set
        label = set_labels[current_set_pos] if current_set_pos < len(set_labels) else f"Session {current_set_pos + 1}"
        emit('set_persona', {"name": label})
        emit('status', {"msg": f"Starting {label} ({current_set_pos + 1}/{set_size})", "color": "magenta"})
    else:
        emit('set_persona', {"name": persona.get('name')})
        emit('status', {"msg": f"Starting Session: {persona.get('name')}", "color": "magenta"})

    tester_context = persona.get("tester_context", "")
    if tester_context:
        emit('show_briefing', {"context": tester_context})
        return  # Wait for tester to click "Ready" before starting the conversation

    _start_conversation(sid, persona, session_configs)


def _start_conversation(sid, persona, session_configs):
    prompt = f"{persona.get('persona', '')}\nGive a very short, 1-sentence opening greeting."
    msg = process_llm([{"role": "developer", "content": prompt}], persona, session_configs)

    active_sessions[sid]["history"].append({"role": "assistant", "content": msg})
    active_sessions[sid]["raw_logs"].append(f"Bot Initial Greeting: {msg}")

    emit('chat', {"speaker": "Bot", "text": msg})
    send_audio(sid, msg, persona, session_configs, is_final=False)


@socketio.on('user_audio_blob')
def handle_user_audio(audio_data):
    sid = request.sid
    session_state = active_sessions.get(sid)
    if not session_state or not session_state["username"]:
        print(f"[{sid[:8]}] WARN: audio received but no session state — prompting client to rejoin")
        emit('unlock_input')
        emit('status', {"msg": "Reconnecting session...", "color": "yellow"})
        emit('request_rejoin')
        return
    if session_state.get("in_survey"):
        emit('unlock_input')  # Still reset client gate so it doesn't freeze
        return

    try:
        session_state["uninterruptible"] = False  # Clear after OVERRIDE turn ends

        cfg = session_state["configs"]
        persona = cfg["personas"][session_state["persona_idx"]]
        session_limit = cfg["session"].get("MAX_TURNS", 5)

        # Instantiate the correct transcription client dynamically
        _, e_client, _ = get_api_clients(cfg.get("secrets", {}))

        temp_file = f"temp_{sid}.webm"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        emit('status', {"msg": "Transcribing...", "color": "yellow"})
        user_text = ""
        try:
            with open(temp_file, "rb") as af:
                if e_client:
                    transcription = e_client.speech_to_text.convert(file=af, model_id="scribe_v2", tag_audio_events=False)
                    user_text = transcription.text
                else:
                    emit('status', {"msg": "No ElevenLabs API Key provided for transcription.", "color": "red"})
        except Exception as e:
            emit('status', {"msg": f"Transcription error: {e}", "color": "red"})
        finally:
            if os.path.exists(temp_file): os.remove(temp_file)

        if not user_text.strip():
            if session_state["last_interruption"]:
                user_text = "*interrupted you but mumbled or trailed off*"
            else:
                emit('status', {"msg": "Listening...", "color": "#666"})
                emit('unlock_input')
                return

        emit('chat', {"speaker": "User", "text": user_text})
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

        llm_prov = persona.get("llm_provider", cfg["model"].get("LLM_PROVIDER", "openai"))
        tts_prov = persona.get("tts_provider", cfg["model"].get("TTS_PROVIDER", "openai"))
        print(f"[{sid[:8]}] LLM={llm_prov} | TTS={tts_prov} | Persona={persona.get('name')}")
        emit('status', {"msg": "Thinking...", "color": "yellow"})
        bot_response = process_llm(messages, persona, cfg)

        found_tags = [t.upper() for t in re.findall(r'\[([a-zA-Z_]+)\]', bot_response)]
        valid_intents = ["BACKCHANNEL", "COOPERATIVE", "COMPETITIVE", "TOPIC_CHANGE", "TERMINATE"]
        valid_strategies = ["RESUME", "BRIDGE", "YIELD", "OVERRIDE", "EXIT"]

        intent = next((t for t in found_tags if t in valid_intents), "UNKNOWN")
        strategy = next((t for t in found_tags if t in valid_strategies), "YIELD")

        clean_response = re.sub(r'\[.*?\]', '', bot_response).strip()
        should_exit = "EXIT" in found_tags or strategy == "EXIT"

        # OVERRIDE behaves like RESUME for text, but also marks the turn as uninterruptible
        if strategy == "OVERRIDE":
            session_state["uninterruptible"] = True
            if session_state.get("remaining_text"):
                clean_response = "..." + session_state["remaining_text"]
        else:
            session_state["uninterruptible"] = False

        if strategy == "RESUME" and session_state.get("remaining_text"):
            clean_response = "..." + session_state["remaining_text"]

        session_state["raw_logs"].append(f"LLM Raw Output: {bot_response}")
        session_state["raw_logs"].append(f"Bot Decision -> Intent: [{intent}] | Strategy: [{strategy}]")
        session_state["raw_logs"].append(f"Bot Spoke: {clean_response}")

        session_state["history"].append({"role": "assistant", "content": bot_response})

        if should_exit or session_state["turn"] >= session_limit:
            session_state["in_survey"] = True
            emit('chat', {"speaker": "Bot", "text": clean_response})

            survey_type = cfg["session"].get("SURVEY_TYPE", "individual")
            set_size = cfg["session"].get("SET_SIZE", 1)

            # Accumulate this persona's data into the set
            session_state["set_dialogues"].append(session_state["history"][:])
            session_state["set_raw_logs"].append(session_state["raw_logs"][:])
            session_state["set_personas"].append(persona.get("name"))

            # Progress is NOT saved here — only saved when survey is submitted
            # If user leaves mid-set, they restart the entire set from Style A

            if survey_type == "comparative" and set_size > 1:
                print(f"[{sid[:8]}] SET CHECK: set_personas={session_state['set_personas']} len={len(session_state['set_personas'])} set_size={set_size}")
                if len(session_state["set_personas"]) >= set_size:
                    # Full set complete - show comparative survey
                    print(f"[{sid[:8]}] SENDING SURVEY after full set completion")
                    try:
                        send_audio(sid, clean_response, persona, cfg, is_final=True, send_survey=True)
                    except Exception as e:
                        print(f"[{sid[:8]}] send_audio FAILED during survey: {e}")
                        _emit_survey_fallback(cfg, session_state)
                else:
                    # Set not complete - show transition screen after audio finishes
                    set_labels = cfg["session"].get("SET_LABELS", [])
                    current_pos = len(session_state["set_personas"])
                    current_label = set_labels[current_pos - 1] if current_pos - 1 < len(set_labels) else f"Session {current_pos}"
                    next_label = set_labels[current_pos] if current_pos < len(set_labels) else f"Session {current_pos + 1}"
                    transition_data = {
                        "completed_label": current_label,
                        "next_label": next_label,
                        "current_pos": current_pos,
                        "set_size": set_size
                    }
                    print(f"[{sid[:8]}] TRANSITION: sending final audio with transition_data={transition_data}")
                    try:
                        send_audio(sid, clean_response, persona, cfg, is_final=True, send_survey=False, transition=transition_data)
                    except Exception as e:
                        print(f"[{sid[:8]}] send_audio FAILED during transition: {e}")
                    # Always emit show_transition as backup — client handles duplicates gracefully
                    emit('show_transition', transition_data)
            else:
                # Individual mode - show survey after each persona
                try:
                    send_audio(sid, clean_response, persona, cfg, is_final=True, send_survey=True)
                except Exception as e:
                    print(f"[{sid[:8]}] send_audio FAILED during individual survey: {e}")
                    _emit_survey_fallback(cfg, session_state)
        else:
            emit('chat', {"speaker": "Bot", "text": clean_response})
            send_audio(sid, clean_response, persona, cfg, is_final=False)
    except Exception as e:
        print(f"[{sid[:8]}] CRITICAL ERROR in handle_user_audio: {e}")
        import traceback; traceback.print_exc()
        emit('status', {"msg": f"Error: {e}", "color": "red"})
        emit('unlock_input')


def _emit_survey_fallback(cfg, session_state):
    """Fallback: directly emit show_survey if send_audio fails."""
    session_cfg = cfg["session"]
    emit('show_survey', {
        "survey": session_cfg.get("SURVEY_QUESTIONS", []),
        "survey_type": session_cfg.get("SURVEY_TYPE", "individual"),
        "set_size": session_cfg.get("SET_SIZE", 1),
        "set_labels": session_cfg.get("SET_LABELS", []),
        "set_personas": session_state.get("set_personas", [])
    })


@socketio.on('interrupt_signal')
def handle_interrupt(data):
    sid = request.sid
    session_state = active_sessions.get(sid)
    if session_state and session_state["is_speaking"] and not session_state.get("in_survey"):
        if session_state.get("uninterruptible"):
            return  # OVERRIDE turn — ignore interrupts
        session_state["is_speaking"] = False
        percent_played = data.get("percent", 0.5)
        full_text = session_state["bot_text"]

        char_idx = int(len(full_text) * percent_played)
        cutoff = full_text[:char_idx]
        remain = full_text[char_idx:]

        interruption_msg = f"SYSTEM NOTE: You were interrupted. You were saying: '{cutoff}'. You intended to say: '{remain}'."
        session_state["last_interruption"] = interruption_msg
        session_state["remaining_text"] = remain
        session_state["raw_logs"].append(f"!!! INTERRUPTION !!! Cutoff: '{cutoff}' | Remaining: '{remain}'")

        emit('status', {"msg": f"Interrupted at: {cutoff}...", "color": "red"})
        emit('stop_audio_playback')


@socketio.on('briefing_ready')
def handle_briefing_ready():
    sid = request.sid
    session_state = active_sessions.get(sid)
    if not session_state or not session_state.get("username"):
        emit('request_rejoin')
        return
    cfg = session_state["configs"]
    persona = cfg["personas"][session_state["persona_idx"]]
    _start_conversation(sid, persona, cfg)


@socketio.on('continue_to_next')
def handle_continue_to_next():
    """Advance to next persona within the current set (mid-set transition)."""
    sid = request.sid
    session_state = active_sessions.get(sid)
    if not session_state or not session_state.get("username"):
        print(f"[{sid[:8]}] WARN: continue_to_next but no session — requesting rejoin")
        emit('request_rejoin')
        return

    cfg = session_state["configs"]
    personas_list = cfg["personas"]
    next_idx = session_state["persona_idx"] + 1

    if next_idx >= len(personas_list):
        emit('set_persona', {"name": "All Sessions Complete"})
        emit('status', {"msg": "All personas completed!", "color": "#00FF00"})
        return

    session_state["persona_idx"] = next_idx
    session_state["history"] = []
    session_state["raw_logs"] = [f"=== STARTING SESSION: {personas_list[next_idx].get('name')} ==="]
    session_state["turn"] = 0
    session_state["last_interruption"] = ""
    session_state["remaining_text"] = ""
    session_state["in_survey"] = False

    persona = personas_list[next_idx]
    set_size = cfg["session"].get("SET_SIZE", 1)
    set_labels = cfg["session"].get("SET_LABELS", [])
    current_set_pos = len(session_state["set_personas"])
    label = set_labels[current_set_pos] if current_set_pos < len(set_labels) else f"Session {current_set_pos + 1}"

    emit('set_persona', {"name": label})
    emit('status', {"msg": f"Starting {label} ({current_set_pos + 1}/{set_size})", "color": "magenta"})

    tester_context = persona.get("tester_context", "")
    if tester_context:
        emit('show_briefing', {"context": tester_context})
        return

    _start_conversation(sid, persona, cfg)


@socketio.on('submit_survey')
def handle_survey(data):
    sid = request.sid
    session_state = active_sessions.get(sid)
    if not session_state or not session_state.get("username"):
        emit('request_rejoin')
        return
    username = session_state["username"]
    cfg = session_state["configs"]
    researcher = cfg["researcher"]
    survey_type = cfg["session"].get("SURVEY_TYPE", "individual")
    set_size = cfg["session"].get("SET_SIZE", 1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if survey_type == "comparative" and set_size > 1:
        # Comparative mode: store all set data together
        set_personas = session_state.get("set_personas", [])
        persona_name = " | ".join(set_personas)

        # Build combined dialogue and logs with per-persona structure
        combined_dialogue = {}
        combined_logs = {}
        set_labels = cfg["session"].get("SET_LABELS", [])
        for i, p_name in enumerate(set_personas):
            label = set_labels[i] if i < len(set_labels) else f"Session {i + 1}"
            combined_dialogue[label] = session_state["set_dialogues"][i] if i < len(session_state["set_dialogues"]) else []
            combined_logs[label] = session_state["set_raw_logs"][i] if i < len(session_state["set_raw_logs"]) else []

        db.save_survey(username, researcher, persona_name, timestamp, combined_dialogue, combined_logs, data)

        # Mark ALL personas in the set as complete
        for p_name in set_personas:
            db.mark_complete(username, p_name)

        # Reset set tracking
        session_state["set_dialogues"] = []
        session_state["set_raw_logs"] = []
        session_state["set_personas"] = []
    else:
        # Individual mode: store single persona data
        persona_name = cfg["personas"][session_state["persona_idx"]].get("name")
        db.save_survey(username, researcher, persona_name, timestamp,
                       session_state["history"], session_state["raw_logs"], data)
        db.mark_complete(username, persona_name)

    emit('status', {"msg": "Survey saved. Loading next persona...", "color": "#00FF00"})
    start_experiment({"username": username})


# --- HELPER FUNCTIONS ---
def process_llm(messages, persona, cfg):
    model_cfg = cfg["model"]
    provider = persona.get("llm_provider", model_cfg.get("LLM_PROVIDER", "openai")).lower()

    # Dynamically grab clients based on researcher keys
    o_client, _, g_client = get_api_clients(cfg.get("secrets", {}))

    try:
        if provider == "openai" and o_client:
            res = o_client.chat.completions.create(model=model_cfg.get("OPENAI_MODEL", "gpt-4o-mini"),
                                                   messages=messages,
                                                   timeout=30)
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


def mark_persona_complete(sid, persona_name):
    """Mark a single persona as complete in the progress table."""
    session_state = active_sessions.get(sid)
    if not session_state:
        return
    username = session_state["username"]
    db.mark_complete(username, persona_name)


def strip_stage_directions(text):
    """Remove stage directions like *(sighs)*, *sighs*, (sighs heavily) before sending to TTS."""
    # Remove *(...)*  or *(...)* patterns
    text = re.sub(r'\*\([^)]*\)\*', '', text)
    # Remove *...* patterns (asterisk-wrapped actions)
    text = re.sub(r'\*[^*]+\*', '', text)
    # Remove standalone (...) that look like stage directions (short, no sentence structure)
    text = re.sub(r'\([^)]{1,50}\)', '', text)
    # Clean up extra whitespace
    text = re.sub(r'  +', ' ', text).strip()
    return text


def send_audio(sid, text, persona, cfg, is_final=False, send_survey=None, transition=None):
    model_cfg = cfg["model"]
    provider = persona.get("tts_provider", model_cfg.get("TTS_PROVIDER", "openai")).lower()

    # Strip *stage directions* but keep *emphasis*
    # Multi-word asterisk = stage direction (*Rolls eyes*, *Sighs heavily*) → remove
    # Capitalized single-word asterisk = stage direction (*Grunts*, *Scoffs*) → remove
    # Lowercase single-word asterisk = emphasis (*my*, *that*, *do*) → keep word
    tts_text = re.sub(r'\*[^*]*\s[^*]*\*', '', text)       # *Rolls eyes* → removed
    tts_text = re.sub(r'\*[A-Z][a-z]*\*', '', tts_text)     # *Grunts* → removed
    tts_text = re.sub(r'\*([^*]+)\*', r'\1', tts_text)      # *my* → my
    tts_text = re.sub(r'  +', ' ', tts_text).strip()
    if not tts_text:
        tts_text = text

    # Dynamically grab clients based on researcher keys
    o_client, e_client, g_client = get_api_clients(cfg.get("secrets", {}))

    audio_data = None
    try:
        if provider == "openai" and o_client:
            res = o_client.audio.speech.create(model=model_cfg.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
                                               voice=persona.get("voice_id",
                                                                 model_cfg.get("OPENAI_DEFAULT_VOICE", "coral")),
                                               input=tts_text, response_format="mp3")
            audio_data = res.content
        elif provider == "elevenlabs" and e_client:
            stream = e_client.text_to_speech.convert(text=tts_text, voice_id=persona.get("voice_id", model_cfg.get(
                "ELEVENLABS_DEFAULT_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")),
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
                model=model_cfg.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"), contents=tts_text, config=conf)

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

    if send_survey is None:
        send_survey = is_final

    if not audio_data:
        print(f"TTS Warning: No audio generated for text: {text[:80]}...")
        socketio.emit('status', {"msg": "TTS failed - no audio generated. Listening...", "color": "red"}, to=sid)
        fail_payload = {"is_final": is_final}
        if send_survey:
            session_cfg = cfg["session"]
            fail_payload["survey"] = session_cfg.get("SURVEY_QUESTIONS", [])
            fail_payload["survey_type"] = session_cfg.get("SURVEY_TYPE", "individual")
            fail_payload["set_size"] = session_cfg.get("SET_SIZE", 1)
            fail_payload["set_labels"] = session_cfg.get("SET_LABELS", [])
            session_state = active_sessions.get(sid)
            if session_state:
                fail_payload["set_personas"] = session_state.get("set_personas", [])
        if transition:
            fail_payload["transition"] = transition
        print(f"[{sid[:8]}] TTS FAILED | emitting tts_failed with send_survey={send_survey}")
        socketio.emit('tts_failed', fail_payload, to=sid)
        return

    print(f"[{sid[:8]}] TTS OK | is_final={is_final} | send_survey={send_survey} | audio_bytes={len(audio_data)}")
    active_sessions[sid]["bot_text"] = text
    active_sessions[sid]["is_speaking"] = True

    session_state = active_sessions.get(sid)
    session_cfg = cfg["session"]
    survey_type = session_cfg.get("SURVEY_TYPE", "individual")
    set_size = session_cfg.get("SET_SIZE", 1)

    payload = {
        "audio_b64": base64.b64encode(audio_data).decode('utf-8'),
        "duration_est": len(text) * 0.05,
        "is_final": is_final,
        "uninterruptible": session_state.get("uninterruptible", False) if session_state else False
    }
    if send_survey:
        payload["survey"] = session_cfg.get("SURVEY_QUESTIONS", [])
        payload["survey_type"] = survey_type
        payload["set_size"] = set_size
        payload["set_labels"] = session_cfg.get("SET_LABELS", [])
        if survey_type == "comparative" and set_size > 1 and session_state:
            payload["set_personas"] = session_state.get("set_personas", [])
    if transition:
        payload["transition"] = transition

    print(f"[{sid[:8]}] Emitting play_audio | survey_in_payload={bool(payload.get('survey'))} | transition_in_payload={bool(payload.get('transition'))} | is_final={is_final}")
    socketio.emit('play_audio', payload, to=sid)
    socketio.emit('status', {"msg": "Speaking...", "color": "cyan"}, to=sid)


if __name__ == '__main__':
    try:
        print("--- Initializing Server on Localhost:5050 ---")
        socketio.run(app, debug=True, host="127.0.0.1", port=5050, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server Stopped.")