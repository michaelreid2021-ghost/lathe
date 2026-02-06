import streamlit as st
import datetime
import time
import os
import re
import yaml
import json
from pathlib import Path

from ai_models import VertexAIModel, LocalModel, GeminiAPIModel, MODEL_TOKEN_LIMITS
from session_manager import SessionContext
from prompt_builder import PromptBuilder
from engine_runner import run_flow
from research_tool import ResearchManager
from skill_manager import SkillManager
import ui_sidebar

# --- Configuration & Prompt Loading ---
PROMPTS_DIR = Path("prompts")
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPTS_PATH = PROMPTS_DIR / "system_prompts.yaml"

# --- DEFINING THE STRUCTURED OUTPUT SCHEMA ---
CHAT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "content": {
            "type": "STRING",
            "description": "The conversational response, explanations, reasoning, and standard chat text. Use Markdown."
        },
        "revisions": {
            "type": "ARRAY",
            "description": "A list of modifications for existing files. Use 'REPLACE' for full file content or 'PATCH' for a diff.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "filename": {
                        "type": "STRING",
                        "description": "The target filename for this code (e.g., 'utils.py')."
                    },
                    "code_content": {
                        "type": "STRING",
                        "description": "The source code or patch content for the file."
                    },
                    "description": {
                        "type": "STRING",
                        "description": "A very brief 3-5 word label for this change (e.g., 'Added error handling')."
                    },
                    "revision_type": {
                        "type": "STRING",
                        "description": "Specify 'REPLACE' for a full file replacement or 'PATCH' for a diff-style patch. If omitted, 'REPLACE' is assumed.",
                        "enum": ["REPLACE", "PATCH"]
                    }
                },
                "required": ["filename", "code_content"]
            }
        },
        "artifacts": {
            "type": "ARRAY",
            "description": "A list of new files to be created. Use this for proposing entirely new artifacts that do not currently exist.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "filename": {
                        "type": "STRING",
                        "description": "The target filename for the new artifact (e.g., 'new_utils.py'). This path should be relative to the project root."
                    },
                    "code_content": {
                        "type": "STRING",
                        "description": "The full source code for the new file."
                    },
                    "description": {
                        "type": "STRING",
                        "description": "A very brief 3-5 word label for this new file."
                    }
                },
                "required": ["filename", "code_content"]
            }
        }
    },
    "required": ["content"]
}

def load_system_prompts() -> dict:
    """Loads system-level prompts from YAML configuration."""
    defaults = {
        "local_summarizer_system": "You are a summarization assistant.",
        "file_summary_template": "Summarize this file:\n{content}",
        "json_repair_template": "Fix this JSON:\n{content}",
        "laconic_summary_template": "Summarize the following exchange:\nUSER: {user_content}\nASSISTANT: {assistant_content}",
        "peer_review_template": "You are a reviewer. Respond in JSON with keys 'approved' (bool) and 'reasoning' (str).",
        "reviewer_memory_entry_template": "- A previously approved response was rejected by the user. Critique: {user_rejection_reason}"
    }
    if SYSTEM_PROMPTS_PATH.exists():
        try:
            prompts = yaml.safe_load(SYSTEM_PROMPTS_PATH.read_text(encoding="utf-8"))
            defaults.update(prompts)
            return defaults
        except Exception as e:
            st.error(f"Failed to load system prompts: {e}")
    return defaults

system_prompts = load_system_prompts()

def format_timedelta(seconds: float) -> str:
    """Formats a duration in seconds into a human-readable string for the AI."""
    if seconds < 1: return "less than a second"
    if seconds < 60: return f"{int(seconds)} second{'s' if int(seconds) != 1 else ''}"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60: return f"{int(minutes)} minute{'s' if int(minutes) != 1 else ''} and {int(seconds)} second{'s' if int(seconds) != 1 else ''}"
    hours, minutes = divmod(minutes, 60)
    if hours < 24: return f"{int(hours)} hour{'s' if int(hours) != 1 else ''} and {int(minutes)} minute{'s' if int(minutes) != 1 else ''}"
    days, hours = divmod(hours, 24)
    return f"{int(days)} day{'s' if int(days) != 1 else ''} and {int(hours)} hour{'s' if int(hours) != 1 else ''}"

def get_most_recent_session() -> Path | None:
    """Finds the most recently modified session directory."""
    sessions_root = Path("chat_sessions")
    if not sessions_root.exists():
        return None
    
    session_dirs = [d for d in sessions_root.iterdir() if d.is_dir()]
    if not session_dirs:
        return None
        
    # Get the directory with the latest modification time
    try:
        latest_session = max(session_dirs, key=lambda p: p.stat().st_mtime)
        return latest_session
    except (OSError, ValueError): # Handle potential race conditions or empty dirs
        return None

def get_session_save_data() -> dict:
    """Constructs the dictionary of the current session state for saving."""
    return {
        "messages": st.session_state.messages,
        "user_persona": st.session_state.user_persona,
        "project_root": st.session_state.get("project_root", os.getcwd()),
        "artifact_context_levels": st.session_state.artifact_context_levels,
        "provisional_context_enabled": st.session_state.provisional_context_enabled,
        "provisional_context_text": st.session_state.provisional_context_text,
        "current_profile": st.session_state.current_profile,
        "instruction_profiles": st.session_state.instruction_profiles,
        "meta_summary": st.session_state.meta_summary,
        "use_laconic_history": st.session_state.use_laconic_history,
        "session_title": st.session_state.session_title,
        "staged_revisions": st.session_state.staged_revisions,
        "reviewer_memory": st.session_state.reviewer_memory,
        "active_skills": st.session_state.active_skills,
        "source_paths": {
            fn: data.get("source_path")
            for fn, data in st.session_state.session_context.artifacts.items()
            if data.get("source_path")
        }
    }

# --- Config ---
st.set_page_config(page_title="Lathe - Vibe-Code Chat Agent", page_icon="🛡️", layout="wide")

# --- Session State Initialization ---
if "session_dir" not in st.session_state:
    most_recent = get_most_recent_session()
    if most_recent:
        st.session_state.session_dir = most_recent
    else:
        # Fallback to creating a new one if no sessions exist
        st.session_state.session_dir = Path(f"chat_sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

if "session_context" not in st.session_state:
    st.session_state.session_context = SessionContext(st.session_state.session_dir)

if "research_manager" not in st.session_state:
    st.session_state.research_manager = ResearchManager(st.session_state.session_dir)

if "skill_manager" not in st.session_state:
    st.session_state.skill_manager = SkillManager()

loaded_state = st.session_state.session_context.load_full_session_state()

# Restore source paths from the main session file if they exist.
if loaded_state.get("source_paths"):
    for filename, source_path in loaded_state["source_paths"].items():
        if filename in st.session_state.session_context.artifacts and source_path:
            st.session_state.session_context.artifacts[filename]["source_path"] = source_path

if "project_root" not in st.session_state:
    st.session_state.project_root = loaded_state.get("project_root", os.getcwd())

default_profile_path = Path("instruction_profiles/Default.yaml")
if default_profile_path.exists():
    try:
        def_data = yaml.safe_load(default_profile_path.read_text(encoding="utf-8"))
        default_persona = def_data.get("persona", "")
    except Exception:
        default_persona = ""
else:
    default_persona = ""

if "session_title" not in st.session_state: st.session_state.session_title = loaded_state.get("session_title", f"Session from {st.session_state.session_dir.name}")
if "messages" not in st.session_state: st.session_state.messages = loaded_state.get("messages", []) or st.session_state.session_context.load_chat_history()
if "user_persona" not in st.session_state: st.session_state.user_persona = loaded_state.get("user_persona", default_persona)
if "request_cancelled" not in st.session_state: st.session_state.request_cancelled = False
if "last_interaction_time" not in st.session_state: st.session_state.last_interaction_time = time.time()
if "artifact_context_levels" not in st.session_state: st.session_state.artifact_context_levels = loaded_state.get("artifact_context_levels", {})
if "provisional_context_enabled" not in st.session_state: st.session_state.provisional_context_enabled = loaded_state.get("provisional_context_enabled", False)
if "provisional_context_text" not in st.session_state: st.session_state.provisional_context_text = loaded_state.get("provisional_context_text", "")
if "instruction_profiles" not in st.session_state: st.session_state.instruction_profiles = loaded_state.get("instruction_profiles", {})
if "current_profile" not in st.session_state: st.session_state.current_profile = loaded_state.get("current_profile", "Default")
if "run_history_compression" not in st.session_state: st.session_state.run_history_compression = False
if "meta_summary" not in st.session_state: st.session_state.meta_summary = loaded_state.get("meta_summary", "")
if "use_laconic_history" not in st.session_state: st.session_state.use_laconic_history = loaded_state.get("use_laconic_history", False)
if "staged_revisions" not in st.session_state: st.session_state.staged_revisions = loaded_state.get("staged_revisions", {})
if "reviewer_memory" not in st.session_state: st.session_state.reviewer_memory = loaded_state.get("reviewer_memory", [])
if "active_skills" not in st.session_state: st.session_state.active_skills = loaded_state.get("active_skills", [])

# New session state for full screen sidebar mode
if "full_screen_sidebar_mode" not in st.session_state:
    st.session_state.full_screen_sidebar_mode = False

if "profiles_dir" not in st.session_state:
    st.session_state.profiles_dir = Path("instruction_profiles")
    st.session_state.profiles_dir.mkdir(exist_ok=True)
    for profile_file in st.session_state.profiles_dir.glob("*.yaml"):
        with open(profile_file, 'r', encoding="utf-8") as f:
            profile_data = yaml.safe_load(f)
            st.session_state.instruction_profiles[profile_file.stem] = profile_data

if st.session_state.current_profile not in st.session_state.instruction_profiles: st.session_state.current_profile = "Default"
if "Default" not in st.session_state.instruction_profiles: st.session_state.instruction_profiles["Default"] = {"persona": default_persona, "provisional_context": ""}

# --- Prompt Flow State ---
if "prompt_flows_dir" not in st.session_state:
    st.session_state.prompt_flows_dir = Path("prompt_flows")
    st.session_state.prompt_flows_dir.mkdir(exist_ok=True)
if "prompt_flows" not in st.session_state:
    st.session_state.prompt_flows = sorted([p.name for p in st.session_state.prompt_flows_dir.glob("*.json") if p.is_file()])
if "use_prompt_flow" not in st.session_state: st.session_state.use_prompt_flow = False
if "selected_prompt_flow" not in st.session_state: st.session_state.selected_prompt_flow = st.session_state.prompt_flows[0] if st.session_state.prompt_flows else ""
if "use_meta_prompting" not in st.session_state: st.session_state.use_meta_prompting = False

# --- Helper: Render Revision Card ---
def render_revision_card(rev_data):
    fn = rev_data.get("filename", "unknown.txt")
    code = rev_data.get("code_content", "")
    desc = rev_data.get("description", "Code Revision")
    action = rev_data.get("action", rev_data.get("revision_type", "REPLACE"))

    with st.container(border=True):
        st.markdown(f"**{fn}** `{action}`")
        st.caption(f"_{desc}_")
        lang = "diff" if action == "PATCH" else "python" # Default to python for CREATE/REPLACE
        st.code(code, language=lang)

# --- Main UI ---
col_title, col_toggle = st.columns([0.8, 0.2])
with col_title:
    st.title("Lathe - Keeping the Vibe alive as you code.")
with col_toggle:
    st.session_state.full_screen_sidebar_mode = st.toggle(
        "Full Screen Sidebar",
        value=st.session_state.full_screen_sidebar_mode,
        help="Expand configuration and management options to fill the main view."
    )

# Determine where sidebar content will be rendered
if st.session_state.full_screen_sidebar_mode:
    sidebar_content_container = st.container()
else:
    sidebar_content_container = st.sidebar

# --- Sidebar Rendering (Model setup must be first) ---
# Get model configuration from the UI, rendered into the chosen container
model_name, debug_mode = ui_sidebar.render_model_config(sidebar_content_container)

now_utc_sidebar = datetime.datetime.now(datetime.timezone.utc)
sidebar_system_instruction = (
    f"Knowledge Cutoff: 01-01-2025\n"
    f"Current Date/time: {now_utc_sidebar.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
    f"{st.session_state.user_persona}\n"
    f"IMPORTANT: You are integrated into a system that expects Structured JSON output."
    f"Use the 'revisions' field for modifications and the 'artifacts' field for new files. For each revision, specify a 'revision_type': 'REPLACE' or 'PATCH'. Use the 'content' field for your conversational response."
)

try:
    # Prioritize API Key if available (Simpler auth)
    #if os.environ.get("GEMINI_API_KEY"):
         #model = GeminiAPIModel(model_name=model_name, system_instruction=sidebar_system_instruction, debug=debug_mode)
    #else:
         # Fallback to Service Account Creds
    model = VertexAIModel(model_name=model_name, system_instruction=sidebar_system_instruction, debug=debug_mode)
    
    local_sys_instruct = system_prompts.get("local_summarizer_system", "You are a helpful summarization assistant.")
    local_summarizer = LocalModel(model_name_override="Local-Summarizer", system_instruction=local_sys_instruct, debug=debug_mode)
    model_ready = True
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    model_ready = False

# Render the rest of the sidebar sections into the chosen container
ui_sidebar.render_all_sidebar_sections(
    parent_container=sidebar_content_container,
    model_ready=model_ready,
    model=model,
    local_summarizer=local_summarizer,
    system_prompts=system_prompts,
    debug_mode=debug_mode,
    sidebar_system_instruction=sidebar_system_instruction
)

# --- Conditional Main Content (Chat History and Input) ---
if not st.session_state.full_screen_sidebar_mode:
    # --- Handle History Compression ---
    if st.session_state.get("run_history_compression") and model_ready:
        with st.spinner("Compressing chat history... This may take a moment."):
            updated = st.session_state.session_context.backfill_laconic_summaries(
                messages=st.session_state.messages,
                local_summarizer=local_summarizer,
                template=system_prompts['laconic_summary_template']
            )
            if updated:
                st.toast("History compression complete!", icon="✅")
                st.session_state.session_context.save_full_session_state(get_session_save_data())
            else:
                st.toast("History is already fully compressed.", icon="ℹ️")
        st.session_state.run_history_compression = False

    # --- Render Chat History ---
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            # Display Text Content
            if 'laconic_content' in msg:
                with st.expander("Show Full Content"):
                     st.markdown(msg["content"])
                st.caption("Laconic Summary:")
                st.markdown(f"> {msg['laconic_content']}")
            else:
                st.markdown(msg["content"])

            # Display Peer Review Result
            if "peer_review_result" in msg and msg["peer_review_result"]:
                review = msg["peer_review_result"]
                approved_val = review.get("approved")
                is_approved = approved_val is True

                reasoning = review.get("reasoning", "No reasoning provided.")
                confidence = review.get("confidence_score", 0.0)

                if approved_val == "error":
                    icon = "❌"
                    title = f"{icon} Peer Review: Failed (Confidence: {confidence:.0%})"
                else:
                    icon = "✅" if is_approved else "⚠️"
                    title = f"{icon} Peer Review: {'Approved' if is_approved else 'Needs Attention'} (Confidence: {confidence:.0%})"

                with st.expander(title):
                    st.markdown(reasoning)

            # Display Structured Revisions (if any)
            if "revisions" in msg and msg["revisions"]:
                st.divider()
                st.caption("📦 Included Revisions")
                
                revisions_in_msg = msg["revisions"]
                for rev in revisions_in_msg:
                    render_revision_card(rev)

                # --- Staging Button for the entire batch ---
                def stage_all_revisions_callback(revisions_to_stage):
                    staged_files = {}
                    has_error = False
                    for rev in revisions_to_stage:
                        filename = rev['filename']
                        action = rev.get('action', rev.get('revision_type', 'REPLACE'))
                        content = rev.get('code_content', '')
                        source_path_str = ""

                        if action == 'CREATE':
                            success, msg, backup_path = st.session_state.session_context.stage_new_file(
                                filename=filename,
                                content=content,
                                project_root=st.session_state.project_root
                            )
                            if success:
                                source_path_str = str(Path(st.session_state.project_root) / filename)
                        else: # REPLACE or PATCH
                            # The revision content needs to be written to a temp file for apply_revisions_to_source to read.
                            st.session_state.session_context.add_revision(filename, content)
                            success, msg, backup_path = st.session_state.session_context.apply_revisions_to_source(filename)
                            if success:
                                source_path_str = st.session_state.session_context.artifacts.get(filename, {}).get('source_path')
                        
                        if success:
                            staged_files[filename] = {"backup_path": backup_path, "action": action, "source_path": source_path_str}
                        else:
                            has_error = True
                            st.error(f"Failed to stage {filename}: {msg}")
                            break # Stop on first error
                    
                    if has_error:
                        st.toast("Staging failed. Rolling back staged changes...", icon="❌")
                        # Rollback any files that were successfully staged in this batch
                        for filename, data in staged_files.items():
                            st.session_state.session_context.revert_candidate(
                                filename=filename, 
                                backup_path_str=data.get('backup_path'),
                                source_path_str=data.get('source_path'),
                                local_model=local_summarizer,
                                summary_template=system_prompts['file_summary_template']
                            )
                    else:
                        st.session_state.staged_revisions = staged_files
                        st.toast("All revisions have been staged for review.", icon="🔬")
                        st.rerun()

                st.button(
                    "Stage All Revisions 🔬", 
                    key=f"stage_all_{msg_idx}",
                    disabled=bool(st.session_state.staged_revisions),
                    on_click=stage_all_revisions_callback,
                    args=(revisions_in_msg,),
                    help=("Another change is already under review. Please accept or reject it first." if bool(st.session_state.staged_revisions) 
                          else "Stage all proposals. New files will be created; existing files will be modified.")
                )

            # Display Execution Logs
            if msg["role"] == "assistant" and "flow_execution_log" in msg:
                with st.expander("Show Flow Execution Details"):
                    for step_log in msg["flow_execution_log"]:
                        step_name = step_log.get("step_name", "Unknown Step")
                        step_type = step_log.get("step_type", "")
                        duration = step_log.get("duration_ms", 0)
                        status_icon = "❌" if "error" in step_log else "✅"

                        st.markdown(f"**{status_icon} {step_name}** (`{step_type}`) - `ran in {duration}ms`")
                        if debug_mode:
                            st.json(step_log)

    prompt = st.chat_input("Ask about your code... (Paste reference tags here)", disabled=not model_ready or bool(st.session_state.staged_revisions))

    # --- Staged Revisions Review Panel (Moved below chat input) ---
    if st.session_state.staged_revisions:
        with st.container(border=True):
            filenames_str = ", ".join([f"`{f}`" for f in st.session_state.staged_revisions.keys()])
            st.subheader(f"🔬 Review Pending Changes for {filenames_str}")
            st.info("The proposed changes have been applied to your local source files. Review the changes and accept or reject the entire batch.")
            
            col1, col2 = st.columns(2)

            def accept_all_callback():
                has_error = False
                for filename, data in st.session_state.staged_revisions.items():
                    success, msg = st.session_state.session_context.commit_candidate(
                        filename=filename,
                        local_model=local_summarizer,
                        summary_template=system_prompts['file_summary_template'],
                        backup_path_str=data.get('backup_path'),
                        source_path_str=data.get('source_path')
                    )
                    if success:
                        st.toast(msg, icon="✅")
                    else:
                        st.error(f"Failed to commit {filename}: {msg}")
                        has_error = True
                
                if not has_error:
                    st.toast("All changes have been committed successfully!", icon="🎉")

                st.session_state.staged_revisions = {}
                st.rerun()

            col1.button("Accept All ✅", use_container_width=True, on_click=accept_all_callback, help="Commit all staged changes, update metadata, and archive originals.")
            
            with col2:
                with st.form(key="rejection_form"):
                    rejection_reason = st.text_area("Reason for Rejection (required):", height=100, placeholder="e.g., The changes introduced a logical bug...")
                    submit_rejection = st.form_submit_button("Reject All & Retry ❌", use_container_width=True)

                    if submit_rejection:
                        if not rejection_reason.strip():
                            st.warning("Please provide a reason for the rejection.")
                        else:
                            # 1. Find the original user prompt that led to these revisions
                            original_prompt = ""
                            last_assistant_message = None
                            for msg in reversed(st.session_state.messages):
                                if msg['role'] == 'assistant':
                                    last_assistant_message = msg
                                if msg['role'] == 'user':
                                    original_prompt = msg['content']
                                    break
                            
                            # 2. Collect all rejected code snippets BEFORE reverting
                            rejected_code_map = {}
                            for filename, data in st.session_state.staged_revisions.items():
                                 try:
                                     rejected_code_map[filename] = Path(data['source_path']).read_text(encoding='utf-8')
                                 except Exception:
                                     rejected_code_map[filename] = "# Failed to read rejected code from source."
                            
                            # 3. Revert all file system changes
                            for filename, data in st.session_state.staged_revisions.items():
                                success, msg = st.session_state.session_context.revert_candidate(
                                    filename=filename,
                                    backup_path_str=data.get('backup_path'),
                                    source_path_str=data.get('source_path'),
                                    local_model=local_summarizer,
                                    summary_template=system_prompts['file_summary_template']
                                )
                                if not success:
                                    st.error(f"CRITICAL: Failed to revert {filename}. Manual intervention may be required. Error: {msg}")

                            # 4. Update the reviewer's memory
                            if last_assistant_message:
                                peer_review_result = last_assistant_message.get("peer_review_result", {})
                                peer_review_reasoning = peer_review_result.get("reasoning", "N/A")
                                memory_entry = system_prompts['reviewer_memory_entry_template'].format(
                                    peer_review_reasoning=peer_review_reasoning,
                                    user_rejection_reason=rejection_reason,
                                    original_prompt=original_prompt,
                                    rejected_files=list(rejected_code_map.keys())
                                )
                                st.session_state.reviewer_memory.append(memory_entry)

                            # 5. Construct and post the comprehensive failure/retry message
                            rejected_files_str = "\n".join(
                                f"**Attempt for `{filename}` (Rejected):**\n```\n{code}\n```"
                                for filename, code in rejected_code_map.items()
                            )
                            failure_context_msg = (
                                f"**REJECTION FEEDBACK**\n\n"
                                f"The previous attempt was rejected. All files have been reverted. Here is the context for your next attempt.\n\n"
                                f"**The Goal (Original Prompt):**\n```\n{original_prompt}\n```\n\n"
                                f"**The Critique (Rejection Reason):**\n> {rejection_reason}\n\n"
                                f"{rejected_files_str}\n\n"
                                f"**Constraint:** Do not attempt the solution(s) found in the rejected attempts again. Please provide a new and corrected implementation."
                            )
                            st.session_state.messages.append({"role": "user", "content": failure_context_msg, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()})
                            
                            st.toast("Changes rejected and reverted. Retry context has been added to the chat.", icon="↩️")
                            st.session_state.staged_revisions = {}
                            st.rerun()

    if prompt:
        st.session_state.request_cancelled = False
        now_utc_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        builder = PromptBuilder(
            session_context=st.session_state.session_context,
            research_manager=st.session_state.research_manager,
            skill_manager=st.session_state.skill_manager,
            active_skills=st.session_state.active_skills,
            artifact_context_levels=st.session_state.artifact_context_levels,
            provisional_context_enabled=st.session_state.provisional_context_enabled,
            provisional_context_text=st.session_state.provisional_context_text,
            last_interaction_time=st.session_state.last_interaction_time,
            format_timedelta=format_timedelta,
            staged_revisions=st.session_state.staged_revisions
        )
        full_prompt_to_ai, display_text = builder.build_prompt(prompt)

        st.session_state.messages.append({"role": "user", "content": display_text, "timestamp": now_utc_iso})
        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_text = ""
            all_proposals = []
            execution_log = None

            if st.button("Cancel Request", key=f"cancel_{len(st.session_state.messages)}", disabled=st.session_state.use_prompt_flow):
                st.session_state.request_cancelled = True
                message_placeholder.markdown("Generation cancelled by user.")
                st.session_state.last_interaction_time = time.time()
            else:
                try:
                    # --- Prompt Flow Execution Path ---
                    if st.session_state.use_prompt_flow and st.session_state.selected_prompt_flow:
                        with st.spinner(f"Executing flow: {st.session_state.selected_prompt_flow}..."):
                            flow_path = st.session_state.prompt_flows_dir / st.session_state.selected_prompt_flow
                            if not flow_path.exists():
                                raise FileNotFoundError(f"Selected prompt flow file not found: {flow_path}")

                            flow_content = json.loads(flow_path.read_text(encoding="utf-8"))

                            history_for_flow_parts = []
                            for msg in st.session_state.messages[:-1]:
                                role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
                                content = msg.get("laconic_content", msg.get("content", ""))
                                history_for_flow_parts.append(f"{role_label}: {content}")

                            initial_context = {
                                "USER_INPUT": full_prompt_to_ai,
                                "CHAT_HISTORY": "\n".join(history_for_flow_parts)
                            }

                            final_output, _step_outputs, execution_log = run_flow(
                                flow_content=flow_content,
                                initial_context=initial_context,
                                default_model=model_name,
                                meta_prompting=st.session_state.use_meta_prompting,
                                debug_mode=debug_mode,
                                verbosity=1 if debug_mode else 0
                            )
                            full_response_text = final_output

                    # --- Standard Chat Path ---
                    else:
                        history_for_model = ""
                        for i, msg in enumerate(st.session_state.messages[:-1]):
                            role_label = "USER" if msg["role"] == "user" else "ASSISTANT"

                            content_to_use = msg.get("laconic_content") if st.session_state.use_laconic_history and msg.get("laconic_content") else msg['content']

                            if msg['role'] == 'user' and "]* " in content_to_use:
                                 content_to_use = content_to_use.split("]* ")[-1]

                            if msg['role'] == 'user':
                                historical_temporal_context = ""
                                if i > 0 and 'timestamp' in msg and 'timestamp' in st.session_state.messages[i-1]:
                                    prev_msg = st.session_state.messages[i-1]
                                    current_ts = datetime.datetime.fromisoformat(msg['timestamp'])
                                    prev_ts = datetime.datetime.fromisoformat(prev_msg['timestamp'])
                                    delta_seconds = (current_ts - prev_ts).total_seconds()
                                    historical_temporal_context = f"current time: {msg['timestamp']}\ntime since last interaction: {format_timedelta(delta_seconds)}.\n"
                                history_for_model += f"{role_label}: {historical_temporal_context}{content_to_use}\n"
                            else:
                                history_for_model += f"{role_label}: {content_to_use}\n"

                        meta_summary_text = ""
                        if st.session_state.meta_summary and st.session_state.meta_summary.strip():
                            meta_summary_text = f"--- OVERALL GOAL/META-SUMMARY ---\n{st.session_state.meta_summary}\n\n"

                        final_input_string = meta_summary_text + history_for_model + f"USER: {full_prompt_to_ai}"

                        with st.spinner("Processing (Structured)..."):
                            if "gemini" in model_name:
                                 response = model.generate(
                                     final_input_string,
                                     response_schema=CHAT_RESPONSE_SCHEMA,
                                     response_mime_type="application/json"
                                 )
                            else:
                                 final_input_string += "\n\nIMPORTANT: Respond in valid JSON format with keys 'content' (string) and 'revisions' (array)."
                                 response = model.generate(final_input_string, response_format={"type": "json_object"})

                            raw_text = response.get('text', '{}')

                            try:
                                response_data = json.loads(raw_text)
                                full_response_text = response_data.get("content", "No content provided.")
                                
                                structured_revisions = response_data.get("revisions", [])
                                new_artifacts = response_data.get("artifacts", [])

                                # Combine into a single list for UI rendering and staging
                                for rev in structured_revisions:
                                    rev['action'] = rev.get('revision_type', 'REPLACE')
                                    all_proposals.append(rev)
                                for art in new_artifacts:
                                    art['action'] = 'CREATE'
                                    all_proposals.append(art)

                            except json.JSONDecodeError:
                                full_response_text = raw_text
                                all_proposals = []
                                st.warning("Model response was not valid JSON. Displaying raw text.")

                    # --- Response Handling --- 
                    if st.session_state.request_cancelled:
                        message_placeholder.markdown("Generation cancelled by user.")
                    else:
                        # --- Peer Review Step ---
                        peer_review_result = None
                        try:
                            user_prompt_for_review = st.session_state.messages[-1]['content']
                            proposed_changes_summary = "\n".join([f"- {p.get('action', 'REPLACE')} {p['filename']}: {p.get('description', 'No description.')}" for p in all_proposals])
                            if not proposed_changes_summary:
                                proposed_changes_summary = "No file changes proposed."

                            reviewer_memory_str = "\n".join(st.session_state.reviewer_memory) if st.session_state.reviewer_memory else "None."

                            review_prompt = system_prompts['peer_review_template'].format(
                                reviewer_memory=reviewer_memory_str,
                                user_prompt=user_prompt_for_review,
                                assistant_content=full_response_text,
                                proposed_changes_summary=proposed_changes_summary
                            )

                            with st.spinner("Performing peer review..."):
                                review_response = local_summarizer.generate(review_prompt, response_format={"type": "json_object"})
                                review_raw_text = review_response.get('text', '{}').strip()
                                if not review_raw_text:
                                    raise ValueError("Local reviewer model returned an empty response.")
                                peer_review_result = json.loads(review_raw_text)

                        except Exception as review_e:
                            st.warning(f"Peer review step failed: {review_e}")
                            peer_review_result = {"approved": "error", "reasoning": f"Failed to perform review: {review_e}", "confidence_score": 0.0}

                        message_placeholder.markdown(full_response_text)

                        # Add revision content to session's revision dir *before* rendering cards
                        if all_proposals:
                            for prop in all_proposals:
                                fn = prop.get("filename")
                                code = prop.get("code_content")
                                if fn and code and prop.get('action') != 'CREATE':
                                    st.session_state.session_context.add_revision(fn, code)

                        assistant_message = {
                            "role": "assistant",
                            "content": full_response_text,
                            "revisions": all_proposals, # Use the combined list
                            "peer_review_result": peer_review_result,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                        if execution_log:
                            assistant_message["flow_execution_log"] = execution_log

                        st.session_state.messages.append(assistant_message)
                        st.session_state.last_interaction_time = time.time()

                        # --- Laconic Summarization ---
                        if full_response_text and full_response_text.strip():
                            try:
                                user_turn_msg = st.session_state.messages[-2]
                                assistant_turn_msg = st.session_state.messages[-1]
                                previous_laconic_summary = "This is the start of the conversation."
                                if len(st.session_state.messages) > 3:
                                    prev_assistant_msg = st.session_state.messages[-3]
                                    previous_laconic_summary = prev_assistant_msg.get("laconic_content", "")

                                laconic_prompt = system_prompts['laconic_summary_template'].format(
                                    previous_laconic_summary=previous_laconic_summary,
                                    user_content=user_turn_msg['content'],
                                    assistant_content=assistant_turn_msg['content']
                                )
                                summary_response = local_summarizer.generate(laconic_prompt)
                                laconic_text = summary_response.get('text', '').strip()
                                if laconic_text:
                                    st.session_state.messages[-1]["laconic_content"] = laconic_text
                            except Exception:
                                pass

                except Exception as e:
                    import traceback
                    st.error(f"An error occurred: {e}")
                    st.code(traceback.format_exc())
        
        # Save session and rerun only after a prompt has been processed.
        st.session_state.session_context.save_full_session_state(get_session_save_data())
        st.rerun()
