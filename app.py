import streamlit as st
import json
import time
import os
from pathlib import Path

# Import your stack
from ai_models import VertexAIModel
from engine_runner import run_flow_file

# --- Config ---
PLAYBOOK_FILE = "decypher_playbook.json"
OUTPUT_DIR = Path("runs_output")
DEBUG_DIR = Path("runs_debug")
OUTPUT_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

st.set_page_config(layout="wide", page_title="Cyderes DeCYpher")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "flow_state" not in st.session_state:
    st.session_state.flow_state = "IDLE" # IDLE, RUNNING, COMPLETED
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "context_memory" not in st.session_state:
    st.session_state.context_memory = ""

# --- Sidebar (Logs) ---
with st.sidebar:
    st.header("⚙️ Live Engine State")
    log_container = st.container()
    if st.button("Reset / New Incident"):
        st.session_state.messages = []
        st.session_state.flow_state = "IDLE"
        st.session_state.current_run_id = None
        st.session_state.context_memory = ""
        st.rerun()

def render_logs():
    if not st.session_state.current_run_id: return
    log_file = DEBUG_DIR / f"{st.session_state.current_run_id}.jsonl"
    if not log_file.exists(): return
    
    # We still read logs for the VISUAL sidebar, but not for the logic
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            try: logs.append(json.loads(line))
            except: pass
    
    with log_container:
        for step in logs:
            step_name = step.get("step_name", "Unknown")
            step_type = step.get("step_type", "Unknown")
            if step_type == "tool_call":
                with st.expander(f"🛠️ {step_name}", expanded=False):
                    st.code(str(step.get("tool_output")), language="json")
            elif step_type == "llm_call":
                with st.expander(f"🧠 {step_name}", expanded=False):
                    st.write(step.get("ai_text"))

def run_playbook(user_prompt):
    st.session_state.flow_state = "RUNNING"
    try:
        with open(PLAYBOOK_FILE, 'r') as f:
            playbook_str = f.read()
        
        # Inject User Input
        clean_input = user_prompt.replace('"', '\\"').replace("\n", " ")
        if "{{USER_INPUT}}" in playbook_str:
            playbook_str = playbook_str.replace("{{USER_INPUT}}", clean_input)
        else:
            playbook_str = playbook_str.replace("{user_input}", clean_input)
        
        run_id = f"run_{int(time.time())}"
        st.session_state.current_run_id = run_id
        temp_path = OUTPUT_DIR / f"{run_id}.json"
        with open(temp_path, "w") as f: f.write(playbook_str)
            
        # --- EXECUTE ENGINE ---
        # Capture the 3rd return value: step_outputs
        case_id, final_output, step_outputs = run_flow_file(
            infile=temp_path,
            outdir=OUTPUT_DIR,
            debugdir=DEBUG_DIR,
            default_model="gemini-2.5-flash",
            verbosity=1,
            debug_mode=True
        )
        
        # --- DIRECT CONTEXT INJECTION ---
        # Convert the dictionary to a string we can feed the Chat LLM
        # This gives it PERFECT memory of every step, input, and output.
        context_str = "SYSTEM CONTEXT [Engine Memory]:\n"
        for step_name, output in step_outputs.items():
            context_str += f"\n--- STEP: {step_name} ---\nOUTPUT: {output}\n"
            
        st.session_state.context_memory = context_str
        st.session_state.flow_state = "COMPLETED"
        return final_output
        
    except Exception as e:
        st.session_state.flow_state = "IDLE"
        return f"Error: {e}"

# --- Main UI ---
st.title("🛡️ DeCYpher: Autonomous SOC Analyst")

# Display Messages (Hide System Prompts)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Enter URL or Threat Info..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Branch 1: Run Flow
    if st.session_state.flow_state == "IDLE":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                final_res = run_playbook(prompt)
                render_logs()
                st.markdown(final_res)
                st.session_state.messages.append({"role": "assistant", "content": final_res})
    
    # Branch 2: Chat with Context
    elif st.session_state.flow_state == "COMPLETED":
        with st.chat_message("assistant"):
            # Construct the prompt using the Dictionary we saved in memory
            chat_context = f"{st.session_state.context_memory}\n\nCHAT HISTORY:\n"
            for m in st.session_state.messages:
                if m["role"] != "system":
                    chat_context += f"{m['role'].upper()}: {m['content']}\n"
            
            # Simple Chat Generation
            model = VertexAIModel("gemini-2.5-pro", system_instruction="You are a helpful security analyst. Use the [Engine Memory] to answer questions truthfully.")
            resp = model.generate(chat_context + f"\nUSER: {prompt}")
            
            st.markdown(resp['text'])
            st.session_state.messages.append({"role": "assistant", "content": resp['text']})

render_logs()