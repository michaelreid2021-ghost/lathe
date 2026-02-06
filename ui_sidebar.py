import streamlit as st
import datetime
import json
import yaml
import os
from pathlib import Path

# Import types for hinting
from ai_models import VertexAIModel, LocalModel, MODEL_TOKEN_LIMITS
from session_manager import SessionContext
from research_tool import ResearchManager
from skill_manager import SkillManager

def new_chat_callback():
    """Clears session state to start a new chat."""
    # Define keys to reset for a fresh session.
    # We keep things like profiles, flows, and application-level settings.
    keys_to_reset = [
        "session_context",
        "research_manager",
        "messages",
        "user_persona",
        "project_root",
        "artifact_context_levels",
        "provisional_context_enabled",
        "provisional_context_text",
        "current_profile",
        "meta_summary",
        "use_laconic_history",
        "session_title",
        "staged_revisions",
        "last_interaction_time",
        "run_history_compression",
        "request_cancelled",
        "active_skills",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    # Create and set a new session directory, which will trigger
    # a full re-initialization on the next run.
    st.session_state.session_dir = Path(f"chat_sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    st.rerun()

def render_model_config(parent_container) -> tuple[str, bool]:
    """Renders model selection and debug mode toggle into the specified container."""
    with parent_container:
        st.header("⚙️ Configuration")

        # --- Session Title Editor ---
        st.session_state.session_title = st.text_input(
            "Session Title",
            value=st.session_state.session_title,
            help="A friendly name for this chat session. Saved automatically."
        )
        st.divider()

        model_name = st.selectbox("Select Model", list(MODEL_TOKEN_LIMITS.keys()), index=0)
        debug_mode = st.toggle("Enable Debug Mode", value=False, help="If enabled, saves raw prompt to 'raw_debug_folder'.")
    return model_name, debug_mode

def render_project_settings():
    """Allows user to define the local project root for smarter file resolution."""
    st.header("TB Project Settings")
    
    # Default to current working directory, but allow override
    default_root = os.getcwd()
    if "project_root" not in st.session_state:
        st.session_state.project_root = default_root

    new_root = st.text_input(
        "Local Project Root", 
        value=st.session_state.project_root,
        help="The root directory where your source code lives. Used to auto-link uploaded files to their source."
    )
    
    if new_root != st.session_state.project_root:
        st.session_state.project_root = new_root
        st.toast(f"Project root updated: {new_root}", icon="📂")

def render_prompt_flow_selector():
    """Renders UI for selecting and enabling prompt flows."""
    st.header("⚡ Prompt Flows")

    if not st.session_state.get("prompt_flows"):
        st.info("No prompt flows found in the 'prompt_flows' directory.")
        return

    st.toggle("Use Prompt Flow", key="use_prompt_flow", help="If enabled, the user prompt will be used as input to the selected flow instead of a direct chat.")

    if st.session_state.use_prompt_flow:
        st.selectbox(
            "Select Prompt Flow",
            options=st.session_state.prompt_flows,
            key="selected_prompt_flow"
        )
        st.checkbox(
            "Use Meta-Prompting (Tasking Brief)",
            key="use_meta_prompting",
            help="Adds a detailed tasking brief to the system prompt for each LLM step in the flow."
        )

def render_session_management(model: VertexAIModel, system_prompts: dict):
    """Renders session download, upload, and switching UI."""
    st.subheader("💾 Session Management")
    st.caption(f"Current Session ID: `{st.session_state.session_dir.name}`")

    # --- New Chat Button ---
    st.button("➕ New Chat", on_click=new_chat_callback, use_container_width=True, help="Start a new, empty chat session.")

    # --- 1. Session Switching (The "Move To" Logic) ---
    sessions_root = Path("chat_sessions")
    if sessions_root.exists():
        existing_sessions = sorted(
            [d for d in sessions_root.iterdir() if d.is_dir()],
            key=lambda x: x.name, 
            reverse=True
        )
        
        # Create a mapping from a display name to the session's directory name (ID)
        session_options = {}
        for s_path in existing_sessions:
            display_name = s_path.name
            state_file = s_path / "full_session_state.json"
            if state_file.exists():
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    title = data.get("session_title")
                    if title and title.strip():
                        # Prepend title to the ID for a user-friendly display
                        display_name = f"{title} ({s_path.name})"
                except (json.JSONDecodeError, IOError):
                    # If file is corrupt or unreadable, fall back to just the directory name
                    pass
            session_options[display_name] = s_path.name
        
        # Use the display names for the selectbox options
        selectable_options = list(session_options.keys())
        
        selected_display_name = st.selectbox(
            "Switch to Existing Session", 
            ["(Select to Switch)"] + selectable_options,
            index=0
        )

        if selected_display_name != "(Select to Switch)":
            # Map the selected display name back to the actual directory name
            selected_session_name = session_options[selected_display_name]
            target_dir = sessions_root / selected_session_name
            
            is_fresh = len(st.session_state.messages) == 0
            
            btn_label = "📂 Open Session" if is_fresh else "⚠️ Abandon Current & Open"
            
            if st.button(btn_label, type="primary"):
                # Define a list of keys to clear from session state on switch.
                # This forces the main app's initialization logic to re-run for these keys.
                keys_to_reset = [
                    "session_context",
                    "research_manager",
                    "messages",
                    "user_persona",
                    "artifact_context_levels",
                    "provisional_context_enabled",
                    "provisional_context_text",
                    "current_profile",
                    "meta_summary",
                    "use_laconic_history",
                    "session_title",
                    "staged_revisions",
                    "active_skills"
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set the new session directory
                st.session_state.session_dir = target_dir
                
                # Rerun to let the main app fully re-initialize from the new directory
                st.rerun()

    st.divider()

    # --- 2. Manual Export/Import (Legacy/Backup) ---
    session_data = {
        "user_persona": st.session_state.user_persona,
        "messages": st.session_state.messages,
        "artifact_context_levels": st.session_state.artifact_context_levels,
        "provisional_context_enabled": st.session_state.provisional_context_enabled,
        "provisional_context_text": st.session_state.provisional_context_text,
        "current_profile": st.session_state.current_profile,
        "instruction_profiles": st.session_state.instruction_profiles,
        "meta_summary": st.session_state.meta_summary,
        "use_laconic_history": st.session_state.use_laconic_history,
        "active_skills": st.session_state.active_skills,
        "source_paths": {fn: data.get("source_path") for fn, data in st.session_state.session_context.artifacts.items() if data.get("source_path")}
    }
    session_json = json.dumps(session_data, indent=2)
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="Download JSON Snapshot",
        data=session_json,
        file_name=f"chat_session_{time_str}.json",
        mime="application/json"
    )

    uploaded_session = st.file_uploader("Import JSON Snapshot", type=["json"], key="session_loader")
    if uploaded_session is not None:
        try:
            loaded_data = json.load(uploaded_session)
            if isinstance(loaded_data, dict) and "messages" in loaded_data:
                st.info("Snapshot Loaded.")
                if st.button("✅ Apply Snapshot to Current Session", type="primary"):
                    st.session_state.user_persona = loaded_data.get("user_persona", st.session_state.user_persona)
                    st.session_state.messages = loaded_data["messages"]
                    st.session_state.artifact_context_levels = loaded_data.get("artifact_context_levels", {})
                    st.session_state.provisional_context_enabled = loaded_data.get("provisional_context_enabled", False)
                    st.session_state.provisional_context_text = loaded_data.get("provisional_context_text", "")
                    st.session_state.current_profile = loaded_data.get("current_profile", "Default")
                    st.session_state.instruction_profiles = loaded_data.get("instruction_profiles", {})
                    st.session_state.meta_summary = loaded_data.get("meta_summary", "")
                    st.session_state.use_laconic_history = loaded_data.get("use_laconic_history", False)
                    st.session_state.active_skills = loaded_data.get("active_skills", [])

                    if "source_paths" in loaded_data:
                        for fn, sp in loaded_data["source_paths"].items():
                            if fn in st.session_state.session_context.artifacts:
                                st.session_state.session_context.artifacts[fn]["source_path"] = sp

                    st.success("Snapshot applied! Rerunning...")
                    st.rerun()
        except Exception as e:
            st.error(f"Error reading JSON: {e}")

def render_artifact_manager(local_summarizer: LocalModel, system_prompts: dict):
    """Renders the file upload and artifact management UI."""
    st.subheader("📄 Attach & Manage Artifacts")
    
    proj_root = st.session_state.get("project_root", os.getcwd())
    st.caption(f"Resolving sources relative to: `{proj_root}`")

    # --- Manual File Creation ---
    with st.form("new_artifact_form", clear_on_submit=True):
        st.markdown("##### Create New File")
        new_filename = st.text_input("Filename (relative to project root)", placeholder="e.g., src/new_module.py", label_visibility="collapsed")
        submitted = st.form_submit_button("➕ Create & Add")
        if submitted and new_filename:
            success, msg = st.session_state.session_context.create_empty_artifact(
                filename=new_filename,
                project_root=proj_root,
                local_model=local_summarizer,
                summary_template=system_prompts.get("file_summary_template", "Summarize: {content}")
            )
            if success:
                st.toast(msg, icon="✅")
                st.rerun()
            else:
                st.error(msg)
    st.divider()

    # --- File Uploader ---
    uploaded_files = st.file_uploader("Upload files", type=["html","txt", "py", "json", "md", "csv", "log", "yaml", "jsonl"], accept_multiple_files=True)
    new_files = [f for f in uploaded_files if f.name not in st.session_state.session_context.artifacts] if uploaded_files else []
    
    if new_files:
        with st.spinner(f"Analyzing {len(new_files)} new file(s)..."):
            for f in new_files:
                st.session_state.session_context.add_file(
                    f, 
                    local_summarizer, 
                    system_prompts.get("file_summary_template", "{content}"),
                    project_root=proj_root
                )
        st.rerun()

    if st.session_state.session_context.artifacts:
        with st.expander("Managed Artifacts", expanded=True):
            def _update_source_path(filename_to_update):
                new_path = st.session_state[f"source_{filename_to_update}"]
                st.session_state.session_context.set_source_path(filename_to_update, new_path if new_path.strip() else None)

            for filename in sorted(st.session_state.session_context.artifacts.keys()):
                data = st.session_state.session_context.artifacts[filename]
                with st.container(border=True):
                    st.markdown(f"**`{filename}`**")

                    level = st.session_state.artifact_context_levels.get(filename, "Background")
                    tag_map = {"Focus": "focal", "Dependency": "dependency", "Background": "background", "Exclude": "exclude"}
                    tag_name = tag_map.get(level, "background")
                    clipboard_payload = f'<{{tag_name}} name="{filename}"/>'
                    st.code(clipboard_payload, language="xml")

                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col1:
                        st.text_input(
                            "Source Path",
                            value=data.get("source_path", ""),
                            key=f"source_{filename}",
                            on_change=_update_source_path,
                            args=(filename,),
                            label_visibility="collapsed",
                            placeholder="File path..."
                        )

                    with col2:
                        if st.button("Pull", key=f"syncfrom_{filename}", use_container_width=True, help="Pull latest content FROM the source path and refresh metadata."):
                            success, msg = st.session_state.session_context.sync_from_source(
                                filename,
                                local_summarizer,
                                system_prompts.get("file_summary_template", "{content}")
                            )
                            if success: 
                                st.toast(msg, icon="✅")
                                st.rerun()
                            else: 
                                st.error(msg)
                    with col3:
                        # This button's functionality is handled in the main chat UI via revision cards
                        st.button("Push", key=f"applyto_{filename}", use_container_width=True, help="'Push' is handled by generating and staging a revision in the chat.", disabled=True)

                    level_options = ["Background", "Dependency", "Focus", "Exclude"]
                    current_level_idx = level_options.index(st.session_state.artifact_context_levels.get(filename, "Background"))
                    st.radio("Level:", level_options, index=current_level_idx, key=f"level_{filename}", horizontal=True, label_visibility="collapsed")
                    if f"level_{filename}" in st.session_state:
                        st.session_state.artifact_context_levels[filename] = st.session_state[f"level_{filename}"]

def render_research_manager(local_summarizer: LocalModel, system_prompts: dict):
    """Renders the UI for managing web research sources."""
    st.subheader("🌐 Research Sources")
    research_manager: ResearchManager = st.session_state.research_manager

    with st.form("add_url_form", clear_on_submit=True):
        url_input = st.text_input("Add URL for research", placeholder="https://example.com/article")
        submitted = st.form_submit_button("Fetch & Add")
        if submitted and url_input:
            with st.spinner(f"Fetching {url_input}..."):
                success, msg = research_manager.add_url(
                    url_input,
                    local_summarizer,
                    system_prompts.get("file_summary_template", "Summarize: {content}")
                )
            if success:
                st.toast(msg, icon="✅")
            else:
                st.error(msg)
            # No need to rerun here, the form submission will trigger it.

    if research_manager.sources:
        with st.expander("Managed Sources", expanded=True):
            sorted_sources = sorted(research_manager.sources.items(), key=lambda item: item[0])
            for url, data in sorted_sources:
                with st.container(border=True):
                    st.markdown(f"**URL:** `{url}`")
                    st.caption(f"Status: {data.get('status', 'Unknown')}")
                    
                    if data.get('summary'):
                        with st.popover("Show Summary"):
                            st.markdown(data['summary'])

                    c1, c2 = st.columns([0.7, 0.3])
                    with c1:
                        def toggle_callback(url_to_toggle):
                            new_state = st.session_state[f"toggle_{url_to_toggle}"]
                            research_manager.toggle_source(url_to_toggle, new_state)

                        st.toggle(
                            "Include in Context",
                            value=data.get("enabled", False),
                            key=f"toggle_{url}",
                            on_change=toggle_callback,
                            args=(url,)
                        )
                    with c2:
                        def remove_callback(url_to_remove):
                            research_manager.remove_source(url_to_remove)
                            st.rerun()

                        st.button("Remove", key=f"remove_{url}", on_click=remove_callback, args=(url,), use_container_width=True)

def render_skill_manager():
    """Renders the UI for selecting Antigravity skills."""
    st.subheader("🌌 Antigravity Skills")
    skill_manager: SkillManager = st.session_state.skill_manager

    if not skill_manager.skills_found:
        st.warning("Antigravity skills repository not found. Please run:\n\n`git clone https://github.com/sickn33/antigravity-awesome-skills.git antigravity_skills`")
        return

    all_skill_options = []
    for category, skills in skill_manager.get_all_skills().items():
        for skill_id, skill_data in skills.items():
            # Format for display: [Category] Skill Name
            display_name = f"[{category}] {skill_data['name']}"
            all_skill_options.append(display_name)

    # We store the display name in the session state
    st.session_state.active_skills = st.multiselect(
        "Activate Skills for Next Prompt",
        options=sorted(all_skill_options),
        default=st.session_state.get('active_skills', []),
        help="Select skills to be injected into the context for the next AI generation."
    )

def render_snippet_manager():
    """Renders the UI for captured code snippets."""
    if st.session_state.session_context.snippets:
        with st.expander("Captured Snippets", expanded=False):
            sorted_snippets = sorted(st.session_state.session_context.snippets.items(), reverse=True)
            for snippet_name, data in sorted_snippets:
                with st.container(border=True):
                    st.markdown(f"**`{snippet_name}`**")
                    clipboard_payload = f'<snippet name="{snippet_name}"/>'
                    st.code(clipboard_payload, language="xml")

def render_provisional_context():
    """Renders the UI for enabling and editing provisional context."""
    st.subheader("🛠️ Provisional Context")
    st.session_state.provisional_context_enabled = st.toggle("Enable Provisional Context", value=st.session_state.provisional_context_enabled)
    if st.session_state.provisional_context_enabled:
        st.session_state.provisional_context_text = st.text_area("Post-Instruction Text", value=st.session_state.provisional_context_text, height=100)

def render_instruction_profiles():
    """Renders the UI for selecting, creating, and editing instruction profiles."""
    st.subheader("📋 Instruction Profiles")
    profile_names = list(st.session_state.instruction_profiles.keys()) + ["New Profile"]
    current_profile_in_list = st.session_state.current_profile in profile_names
    current_profile_idx = profile_names.index(st.session_state.current_profile) if current_profile_in_list else 0
    selected_profile = st.selectbox("Select Profile", profile_names, index=current_profile_idx)

    if selected_profile == "New Profile":
        new_profile_name = st.text_input("New Profile Name")
        if st.button("Create Profile") and new_profile_name:
            st.session_state.instruction_profiles[new_profile_name] = {"persona": st.session_state.user_persona, "provisional_context": st.session_state.provisional_context_text}
            with open(st.session_state.profiles_dir / f"{new_profile_name}.yaml", 'w', encoding="utf-8") as f:
                yaml.dump(st.session_state.instruction_profiles[new_profile_name], f)
            st.session_state.current_profile = new_profile_name
            st.rerun()
    elif selected_profile:
        if selected_profile != st.session_state.current_profile:
            profile_data = st.session_state.instruction_profiles[selected_profile]
            st.session_state.user_persona = profile_data.get("persona", "")
            st.session_state.provisional_context_text = profile_data.get("provisional_context", "")
            st.session_state.current_profile = selected_profile
            st.rerun()

        profile_data = st.session_state.instruction_profiles[st.session_state.current_profile]
        persona_val = st.text_area("Persona (Editable)", value=profile_data.get("persona", ""), height=100)
        prov_ctx_val = st.text_area("Provisional Context (Editable)", value=profile_data.get("provisional_context", ""), height=100)

        st.session_state.user_persona = persona_val
        st.session_state.provisional_context_text = prov_ctx_val

        if st.button("Save Changes to Profile"):
            st.session_state.instruction_profiles[st.session_state.current_profile] = {"persona": persona_val, "provisional_context": prov_ctx_val}
            with open(st.session_state.profiles_dir / f"{st.session_state.current_profile}.yaml", 'w', encoding="utf-8") as f:
                yaml.dump(st.session_state.instruction_profiles[st.session_state.current_profile], f)
            st.success("Profile saved!")

def render_token_usage(model: VertexAIModel, system_instruction: str):
    """Renders the token usage statistics for the current context."""
    st.subheader("📊 Token Usage")
    with st.expander("Show Context Token Count", expanded=True):
        with st.spinner("Calculating accurate tokens..."):
            system_prompt_tokens = model.count_tokens(system_instruction)
            context_str_for_count = st.session_state.session_context.get_context_string(st.session_state.artifact_context_levels)
            files_tokens = model.count_tokens(context_str_for_count)
            provisional_tokens = model.count_tokens(st.session_state.provisional_context_text) if st.session_state.provisional_context_enabled else 0

            history_str_for_counting = "\n".join([msg.get("laconic_content", msg["content"]) for msg in st.session_state.messages])
            history_tokens = model.count_tokens(history_str_for_counting)

            total_tokens = system_prompt_tokens + files_tokens + history_tokens + provisional_tokens

        st.text(f"System Prompt: {system_prompt_tokens:,} tokens")
        st.text(f"Chat History: {history_tokens:,} tokens")
        st.text(f"Attached Artifacts: {files_tokens:,} tokens")
        st.text(f"Provisional Context: {provisional_tokens:,} tokens")
        st.info(f"**Total Context Tokens: {total_tokens:,} / {model.token_limit:,}**")

        if total_tokens > model.token_limit:
            st.error("Context limit exceeded. Please clear history or adjust artifacts.")
        elif total_tokens > model.token_limit * 0.9:
            st.warning("Context is approaching the token limit.")

def render_system_controls():
    """Renders final system-level controls like persona and history clear."""
    st.subheader("System & History Controls")

    st.session_state.meta_summary = st.text_area(
        "Pinned Meta-Summary",
        value=st.session_state.get("meta_summary", ""),
        height=120,
        help="A pinned, high-level summary of the conversation's goals. This is always included at the top of the AI's context window."
    )

    st.session_state.use_laconic_history = st.toggle(
        "Use Laconic History",
        value=st.session_state.get("use_laconic_history", False),
        help="If enabled, sends the condensed 'laconic' summaries of past turns to the model instead of the full text, saving tokens."
    )

    st.session_state.user_persona = st.text_area("Persona", value=st.session_state.user_persona, height=200)

    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.session_state.request_cancelled = False
        st.rerun()

    if st.button("Backfill History Summaries", help="Generate laconic summaries for any turns in the history that are missing them."):
        st.session_state.run_history_compression = True
        st.rerun()

def render_all_sidebar_sections(parent_container, model_ready: bool, model, local_summarizer, system_prompts: dict, debug_mode: bool, sidebar_system_instruction: str):
    """Renders all other sidebar sections into the specified container, using a multi-column layout if not in the native sidebar."""
    # Check if we are in full-screen mode by seeing if the parent is the main container, not the sidebar object.
    is_fullscreen = parent_container is not st.sidebar

    with parent_container:
        if is_fullscreen:
            # Multi-column layout for the main page view
            col1, col2, col3 = st.columns([1, 1.2, 1])

            with col1:
                render_project_settings()
                st.divider()
                
                if model_ready:
                    render_session_management(model, system_prompts)
                else:
                    st.warning("Session management disabled until model is ready.")
                
                st.divider()
                render_prompt_flow_selector()
                
            with col2:
                if model_ready:
                    render_artifact_manager(local_summarizer, system_prompts)
                    st.divider()
                    render_research_manager(local_summarizer, system_prompts)
                    st.divider()
                    render_skill_manager()
                    st.divider()
                    render_snippet_manager()
                else:
                    st.warning("Artifact & Research management disabled until model is ready.")

            with col3:
                render_provisional_context()
                st.divider()
                render_instruction_profiles()
                st.divider()
                
                if model_ready:
                    render_token_usage(model, sidebar_system_instruction)
                else:
                    st.warning("Token usage cannot be calculated as model is not ready.")
                
                st.divider()
                render_system_controls()

        else:
            # Original sequential layout for the native sidebar
            render_project_settings()

            st.divider()
            if model_ready:
                render_session_management(model, system_prompts)
            else:
                st.warning("Session management disabled until model is ready.")

            st.divider()
            if model_ready:
                render_artifact_manager(local_summarizer, system_prompts)
                st.divider()
                render_research_manager(local_summarizer, system_prompts)
                st.divider()
                render_skill_manager()
                st.divider()
                render_snippet_manager()
            else:
                st.warning("Artifact & Research management disabled until model is ready.")

            st.divider()
            render_provisional_context()

            st.divider()
            render_prompt_flow_selector()

            st.divider()
            render_instruction_profiles()

            st.divider()
            if model_ready:
                render_token_usage(model, sidebar_system_instruction)
            else:
                st.warning("Token usage cannot be calculated as model is not ready.")

            st.divider()
            render_system_controls()
