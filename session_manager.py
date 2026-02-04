import datetime
import shutil
import json
import ast
import re
import os
from pathlib import Path

# Import type hints from the models module without creating a circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ai_models import VertexAIModel, LocalModel


class SessionContext:
    """Manages all artifacts for a single chat session, including context levels, chat history, and source sync."""
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.artifacts_dir = self.session_dir / "artifacts"
        self.originals_dir = self.artifacts_dir / "originals"
        self.summaries_dir = self.artifacts_dir / "summaries"
        self.skeletons_dir = self.artifacts_dir / "skeletons"
        self.revisions_dir = self.artifacts_dir / "revisions"
        self.applied_revisions_dir = self.revisions_dir / "applied"
        self.rejected_revisions_dir = self.revisions_dir / "rejected"
        self.archive_dir = self.session_dir / "archive" # For approved original files
        self.snippets_dir = self.session_dir / "snippets"
        self.chat_history_path = self.session_dir / "chat_history.json"
        self.full_session_state_path = self.session_dir / "full_session_state.json"
        self.artifacts = {}  # filename -> {"summary": str, "path": Path, "skeleton_path": Path, "source_path": str or None}
        self.snippets = {}   # snippet_filename -> {"path": Path, "language": str}
        self._create_dirs()
        self._load_existing_artifacts()
        self._load_existing_snippets()

    def _create_dirs(self):
        """Creates the necessary directory structure for the session."""
        for d in [self.originals_dir, self.summaries_dir, self.skeletons_dir, self.revisions_dir, self.applied_revisions_dir, self.rejected_revisions_dir, self.archive_dir, self.snippets_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _update_artifact_metadata(self, filename: str, content: str, local_model: 'LocalModel', summary_template: str):
        """
        Generates and saves new summary and skeleton for a given file content.
        Updates the in-memory artifact dictionary. This is a core function for ensuring
        session state is consistent with file content.
        """
        # Generate and save new summary
        summary_prompt = summary_template.format(filename=filename, content=content)
        summary = local_model.generate(summary_prompt).get("text", "Error summarizing file.")
        summary_path = self.summaries_dir / f"{filename}.summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        # Generate and save new skeleton
        skeleton = self._generate_skeleton(content, filename)
        skeleton_path = self.skeletons_dir / f"{filename}.skeleton.txt"
        skeleton_path.write_text(skeleton, encoding="utf-8")

        # Update in-memory artifact data
        if filename in self.artifacts:
            self.artifacts[filename]["summary"] = summary
            # Ensure skeleton_path is correctly set if it was missing
            if "skeleton_path" not in self.artifacts[filename] or not self.artifacts[filename]["skeleton_path"]:
                 self.artifacts[filename]["skeleton_path"] = skeleton_path

    def _generate_skeleton(self, source_code: str, filename: str) -> str:
        """Generates a structural skeleton of a Python file using AST."""
        if not filename.endswith('.py'):
            return f"# --- SKELETON FOR {filename} ---\n# Non-Python file, no skeleton available.\n"
        try:
            tree = ast.parse(source_code)
            skeleton = [f"# SKELETON of {filename}"]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names: skeleton.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    names = ', '.join(alias.name for alias in node.names)
                    skeleton.append(f"from {'.' * node.level}{module} import {names}")
                elif isinstance(node, ast.ClassDef):
                    skeleton.append(f"\nclass {node.name}:")
                    if not node.body: skeleton.append("    pass")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorator_list = [f"@{ast.unparse(d)}" for d in node.decorator_list]
                    if decorator_list: skeleton.append('\n'.join(decorator_list))
                    signature = f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}def {node.name}{ast.unparse(node.args)} -> {ast.unparse(node.returns) if node.returns else 'None'}"
                    # The 'parent' attribute is not standard in ast.Node, need to walk the tree with parent pointers or use a different approach.
                    # For now, we assume all functions are at a consistent indentation level for the skeleton.
                    is_method = False
                    for parent in ast.walk(tree):
                        if hasattr(parent, 'body') and isinstance(parent.body, list) and node in parent.body and isinstance(parent, ast.ClassDef):
                            is_method = True
                            break
                    if is_method:
                        skeleton.append(f"    {signature}: ...")
                    else:
                        skeleton.append(f"\n{signature}: ...")

            return "\n".join(skeleton)
        except Exception as e:
            return f"# FAILED TO PARSE SKELETON for {filename}: {e}"

    def _load_existing_artifacts(self):
        """Loads file artifacts if the session is being restored."""
        if not self.originals_dir.exists(): return
        for original_file in self.originals_dir.iterdir():
            filename = original_file.name
            summary_path = self.summaries_dir / f"{filename}.summary.txt"
            skeleton_path = self.skeletons_dir / f"{filename}.skeleton.txt"
            if summary_path.exists():
                self.artifacts[filename] = {
                    "summary": summary_path.read_text(encoding="utf-8"),
                    "path": original_file,
                    "skeleton_path": skeleton_path if skeleton_path.exists() else None,
                    "source_path": None # This is populated by the main app from the session state file.
                }

    def _load_existing_snippets(self):
        """Loads code snippets if the session is being restored."""
        if not self.snippets_dir.exists(): return
        for snippet_file in self.snippets_dir.iterdir():
            filename = snippet_file.name
            language = snippet_file.suffix.strip('.')
            self.snippets[filename] = {
                "path": snippet_file,
                "language": language
            }

    def add_file(self, uploaded_file, local_model: 'LocalModel', summary_template: str, project_root: str = None):
        """
        Saves, summarizes, skeletons, and registers a new file artifact.
        
        Args:
            project_root: If provided, attempts to find the file's original path within this directory.
        """
        filename = uploaded_file.name
        content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        original_path = self.originals_dir / filename
        original_path.write_text(content, encoding="utf-8")

        source_path = None
        
        # IMPROVED SOURCE FINDING: Use the specific Project Root if available
        search_root = Path(project_root) if project_root else Path(os.getcwd())
        
        try:
            # We look for an exact name match. This might be slow on huge repos, but it's better than nothing.
            # Using rglob to recursively find.
            found_files = list(search_root.rglob(filename))
            
            # Filter out the session storage itself to avoid self-referencing
            filtered_files = [
                f for f in found_files 
                if "chat_sessions" not in f.parts and ".git" not in f.parts and "__pycache__" not in f.parts
            ]
            
            if filtered_files:
                # Default to the first found, but prefer shortest path (likely closer to root)
                filtered_files.sort(key=lambda p: len(p.parts))
                source_path = str(filtered_files[0].resolve())
        except Exception as e:
            print(f"Error resolving source path for {filename}: {e}")

        # The artifact doesn't exist yet, so we have to create the dict first.
        self.artifacts[filename] = {
            "summary": "", # Will be filled by _update_artifact_metadata
            "path": original_path,
            "skeleton_path": self.skeletons_dir / f"{filename}.skeleton.txt", # Give it the expected path
            "source_path": source_path
        }

        # Now call the centralized updater
        self._update_artifact_metadata(filename, content, local_model, summary_template)
    
    def create_empty_artifact(self, filename: str, project_root: str, local_model: 'LocalModel', summary_template: str):
        """Creates a new, empty file on disk and adds it to the session as an artifact."""
        if not filename.strip():
            return False, "Filename cannot be empty."
        if filename in self.artifacts:
            return False, f"Artifact '{filename}' already exists in the session."

        source_path = Path(project_root) / filename
        if source_path.exists():
            return False, f"File '{source_path}' already exists on disk. Please add it normally."

        try:
            # Create empty file on disk
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.touch()

            # Create empty working copy in the session
            content = ""
            original_path = self.originals_dir / filename
            original_path.write_text(content, encoding="utf-8")

            # Add to the in-memory artifacts dictionary
            self.artifacts[filename] = {
                "summary": "", # Will be generated next
                "path": original_path,
                "skeleton_path": self.skeletons_dir / f"{filename}.skeleton.txt",
                "source_path": str(source_path)
            }

            # Generate and save the initial (empty) metadata
            self._update_artifact_metadata(filename, content, local_model, summary_template)
            
            return True, f"Successfully created and added '{filename}'."
        except Exception as e:
            # Clean up if creation failed
            if source_path.exists():
                source_path.unlink()
            return False, f"Failed to create file: {e}"

    def set_source_path(self, filename: str, source_path: str):
        """Sets the real source path for an artifact in memory. This change is persisted when the main session state is saved."""
        if filename in self.artifacts:
            self.artifacts[filename]["source_path"] = source_path

    def sync_from_source(self, filename: str, local_model: 'LocalModel', summary_template: str):
        """
        Pulls latest content from source path, updates the working copy,
        and regenerates the summary and skeleton to ensure consistency.
        """
        if filename not in self.artifacts:
            return False, "Artifact not found."
        data = self.artifacts[filename]
        source_path_str = data.get("source_path")
        if not source_path_str or not Path(source_path_str).exists():
            return False, "No valid source path set or file does not exist."

        try:
            source_path = Path(source_path_str)
            source_content = source_path.read_text(encoding="utf-8")
            working_path = data["path"]

            backup_path_str = "No backup created (working copy did not exist)."
            if working_path.exists():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{working_path.stem}.backup_{timestamp}{working_path.suffix}"
                backup_path = working_path.parent / backup_name
                shutil.copy2(working_path, backup_path)
                backup_path_str = f"Backup saved as {backup_path.name}"

            working_path.write_text(source_content, encoding="utf-8")

            self._update_artifact_metadata(filename, source_content, local_model, summary_template)

            return True, f"Synced from source and refreshed metadata. {backup_path_str}"
        except Exception as e:
            return False, f"An error occurred during sync: {e}"

    def stage_new_file(self, filename: str, content: str, project_root: str):
        """Stages a new file for creation by writing it to the source tree. No backup is created."""
        if filename in self.artifacts:
            return False, f"Artifact '{filename}' already exists in the session. Use a revision instead.", None
        
        source_path = Path(project_root) / filename
        if source_path.exists():
            return False, f"File '{source_path}' already exists on disk but is not tracked. Aborting to prevent data loss.", None

        try:
            # Ensure parent directory exists
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(content, encoding="utf-8")
            # No backup is created for a new file, so backup_path is None
            return True, f"Successfully staged new file {filename} for review.", None
        except Exception as e:
            return False, f"Failed to stage new file {filename}: {e}", None

    def apply_revisions_to_source(self, filename: str):
        """
        Stages a pending revision for an existing file by applying it to the source file.
        - Creates a backup of the source file.
        - Writes the revision to the live source file.
        """
        if filename not in self.artifacts:
            return False, "Artifact not found in session context.", None
        
        data = self.artifacts[filename]
        source_path_str = data.get("source_path")
        
        if not source_path_str:
            return False, "Source path is not set for this artifact.", None
        
        source_path = Path(source_path_str)
        if not source_path.exists():
            return False, f"Source file does not exist at: {source_path}", None

        pending_revision_files = sorted(self.revisions_dir.glob(f"{filename}.*.patch"))

        if not pending_revision_files:
            return True, "No pending revisions to apply.", None

        backup_path = None
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.name}.backup_{timestamp}"
            backup_path = source_path.parent / backup_name
            shutil.copy2(source_path, backup_path)
            
            latest_revision_path = pending_revision_files[-1]
            new_content = latest_revision_path.read_text(encoding="utf-8")
            
            source_path.write_text(new_content, encoding="utf-8")

            return True, f"Successfully staged revision for {filename}. Ready for review.", str(backup_path)

        except Exception as e:
            if backup_path and backup_path.exists():
                try:
                    shutil.copy2(backup_path, source_path) # Attempt to restore from backup
                    return False, f"An error occurred: {e}. Restored from backup. Staging failed.", None
                except Exception as restore_e:
                     return False, f"CRITICAL ERROR: Staging failed ({e}) AND restore from backup failed ({restore_e}).", None
            return False, f"An error occurred during staging: {e}. No backup was created.", None

    def commit_candidate(self, filename: str, local_model: 'LocalModel', summary_template: str, backup_path_str: str | None):
        """
        Finalizes a staged change. If it's a new file, it's formally added to the session.
        If it's a modification, the backup is archived.
        """
        is_new_file = not backup_path_str

        # For a new file, the source path isn't in artifacts yet. It must be derived.
        # This is brittle. The caller should provide the source_path.
        # For now, we assume commit is called with state from `staged_revisions` which should contain the path.
        # Let's find the source path from artifacts, or assume it's a new file if not found.
        source_path_str = self.artifacts.get(filename, {}).get("source_path")
        if is_new_file and not source_path_str:
             return False, f"Cannot commit new file '{filename}' without a source path."

        if not is_new_file:
            backup_path = Path(backup_path_str)
            if not backup_path.exists():
                 return False, f"Critical error: backup path '{backup_path_str}' not found during commit for {filename}."

        try:
            source_path = Path(source_path_str)
            source_content = source_path.read_text(encoding="utf-8")

            # If it's a new file, create its entry in the session artifacts now.
            if is_new_file:
                working_path = self.originals_dir / filename
                self.artifacts[filename] = {
                    "summary": "", # Will be generated next
                    "path": working_path,
                    "skeleton_path": self.skeletons_dir / f"{filename}.skeleton.txt",
                    "source_path": source_path_str
                }
            
            working_path = self.artifacts[filename]["path"]
            working_path.write_text(source_content, encoding="utf-8")

            self._update_artifact_metadata(filename, source_content, local_model, summary_template)

            if not is_new_file:
                archive_destination = self.archive_dir / backup_path.name
                shutil.move(str(backup_path), str(archive_destination))

            pending_revision_files = sorted(self.revisions_dir.glob(f"{filename}.*.patch"))
            for rev_path in pending_revision_files:
                destination = self.applied_revisions_dir / rev_path.name
                shutil.move(str(rev_path), str(destination))
            
            msg = f"{filename} has been created and added to session." if is_new_file else f"{filename} has been updated and metadata refreshed."
            return True, msg

        except Exception as e:
            return False, f"Failed to commit changes for {filename}. Error: {e}"

    def revert_candidate(self, filename: str, backup_path_str: str | None, source_path_str: str, local_model: 'LocalModel', summary_template: str):
        """Reverts a staged change. Deletes a new file or restores a modified file from backup."""
        is_new_file = not backup_path_str

        if not source_path_str:
            return False, "Source path is missing. Cannot revert."

        try:
            if is_new_file:
                # Reverting a new file means deleting it.
                Path(source_path_str).unlink(missing_ok=True)
                msg = f"Discarded new file proposal for {filename}."
            else:
                # Reverting a modified file means restoring from backup.
                backup_path = Path(backup_path_str)
                if not backup_path.exists():
                    return False, f"Backup path '{backup_path_str}' not found. Cannot revert."
                
                shutil.copy2(backup_path, source_path_str)
                backup_path.unlink() # Clean up the temporary backup file

                # Regenerate metadata from the restored source file to ensure context is correct
                restored_content = Path(source_path_str).read_text(encoding="utf-8")
                self._update_artifact_metadata(filename, restored_content, local_model, summary_template)
                msg = f"Reverted changes to {filename} and discarded candidate."

            # Archive revision files as 'rejected' for both cases.
            pending_revision_files = sorted(self.revisions_dir.glob(f"{filename}.*.patch"))
            for rev_path in pending_revision_files:
                destination = self.rejected_revisions_dir / rev_path.name
                shutil.move(str(rev_path), str(destination))
            
            return True, msg
        except Exception as e:
            return False, f"CRITICAL: Failed to revert {filename}. Error: {e}"

    def add_revision(self, filename: str, code_snippet: str):
        """Adds a revision patch for a file artifact."""
        if filename not in self.artifacts: return None
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clear out old pending revisions for this file to ensure only one is active
        for old_rev in self.revisions_dir.glob(f"{filename}.*.patch"):
            old_rev.unlink()

        revision_path = self.revisions_dir / f"{filename}.{timestamp}.patch"
        revision_path.write_text(code_snippet, encoding="utf-8")
        return revision_path

    def add_code_snippet(self, code_content: str, language_hint: str) -> Path:
        """Saves a captured code snippet from an AI response."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        index = len(list(self.snippets_dir.glob(f"snippet_{timestamp}_*")))

        ext = re.sub(r'[^a-zA-Z0-9]', '', language_hint).lower() if language_hint else "txt"
        if not ext: ext = "txt"

        snippet_filename = f"snippet_{timestamp}_{index}.{ext}"
        snippet_path = self.snippets_dir / snippet_filename
        snippet_path.write_text(code_content, encoding="utf-8")

        self.snippets[snippet_filename] = {"path": snippet_path, "language": ext}
        return snippet_path

    def get_context_string(self, context_levels: dict, manually_inserted_files: list = None, staged_revisions: dict = None) -> str:
        """
        Generates context, omitting files that are already explicitly
        referenced in the user's prompt (manually_inserted_files).
        For staged files, 'Focus' uses the live source code, while other levels
        use the original, pre-stage summary/skeleton.
        """
        if not self.artifacts: return ""
        manually_inserted_files = manually_inserted_files or []
        staged_revisions = staged_revisions or {}
        contexts = []

        for filename, data in self.artifacts.items():
            if filename in manually_inserted_files:
                continue

            level = context_levels.get(filename, "Background")
            if level == "Exclude":
                continue

            is_staged = filename in staged_revisions

            if level == "Focus":
                source_path_str = data.get("source_path")
                content = ""
                # For focus, if a file is staged, we MUST use the live source code.
                if is_staged and source_path_str and Path(source_path_str).exists():
                    content = Path(source_path_str).read_text(encoding="utf-8")
                # If not staged, use the session's stable working copy.
                elif data['path'].exists():
                    content = data['path'].read_text(encoding="utf-8")
                contexts.append(f"--- FOCAL CONTEXT: {filename} ---\n```\n{content}\n```")
            # For Dependency and Background, ALWAYS use the stable, committed metadata, even if staged.
            # This prevents the AI from getting confused by unapproved changes.
            elif level == "Dependency" and data.get('skeleton_path'):
                skeleton_path = data.get('skeleton_path')
                if skeleton_path and Path(skeleton_path).exists():
                    skeleton_content = Path(skeleton_path).read_text(encoding="utf-8")
                    contexts.append(f"--- DEPENDENCY: {filename} ---\n```python\n{skeleton_content}\n```")
            else: # Background level
                summary_content = data.get('summary', '# Summary not available.')
                contexts.append(f"--- BACKGROUND: {filename} ---\n{summary_content}")

        return "\n\n" + "\n\n".join(contexts) + "\n\n"

    def save_chat_history(self, messages: list):
        """Saves the chat history to a JSON file."""
        self.chat_history_path.write_text(json.dumps(messages, indent=2), encoding='utf-8')

    def load_chat_history(self) -> list:
        """Loads the chat history from a JSON file."""
        return json.loads(self.chat_history_path.read_text(encoding='utf-8')) if self.chat_history_path.is_file() else []

    def save_full_session_state(self, session_data: dict):
        """Saves the complete session state to a single JSON file, with a backup of the previous state."""
        prev_state_path = self.full_session_state_path.with_suffix(".json.prev")
        if self.full_session_state_path.exists():
            self.full_session_state_path.replace(prev_state_path)
        self.full_session_state_path.write_text(json.dumps(session_data, indent=2), encoding='utf-8')

    def _repair_session_with_ai(self, raw_content: str, model: 'VertexAIModel', repair_template: str) -> dict:
        """Uses the AI model to reconstruct valid JSON from a corrupted string."""
        repair_prompt = repair_template.format(content=raw_content[:300000])
        try:
            # We use gemini-2.5-pro for high-stakes tasks like this.
            response = model.generate(repair_prompt)
            cleaned_text = response.get('text', '').strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned_text)
        except Exception:
            return {}

    def load_full_session_state(self, model: 'VertexAIModel' = None, repair_template: str = "") -> dict:
        """
        Loads the complete session state with a multi-tiered fallback strategy.
        1. Try Primary state file.
        2. Try Backup (.prev) state file.
        3. If both fail and a model is provided, attempt AI Semantic Repair.
        """
        prev_state_path = self.full_session_state_path.with_suffix(".json.prev")

        for path_to_try in [self.full_session_state_path, prev_state_path]:
            if path_to_try.is_file():
                try:
                    content = path_to_try.read_text(encoding='utf-8')
                    if content:
                        data = json.loads(content)
                        if path_to_try == prev_state_path: # Restore from backup
                           self.full_session_state_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                        return data
                except (json.JSONDecodeError, IOError):
                    continue # Try next file

        if model and repair_template:
            target_file = self.full_session_state_path if self.full_session_state_path.exists() else prev_state_path
            if target_file.exists():
                try:
                    raw_content = target_file.read_text(encoding='utf-8')
                    repaired_data = self._repair_session_with_ai(raw_content, model, repair_template)
                    if repaired_data:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_path = self.full_session_state_path.with_suffix(f".corrupt_{timestamp}")
                        target_file.rename(backup_path)
                        self.full_session_state_path.write_text(json.dumps(repaired_data, indent=2), encoding='utf-8')
                        return repaired_data
                except Exception:
                    pass

        return {}

    def backfill_laconic_summaries(self, messages: list, local_summarizer: 'LocalModel', template: str) -> bool:
        """Iterates through history and generates missing laconic summaries."""
        updated = False
        # Iterate through pairs of user/assistant messages
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i]
            # Ensure we have a valid pair
            if (i + 1) >= len(messages):
                continue
            assistant_msg = messages[i+1]

            # Check if this is a valid turn that needs summarizing
            if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant' and "laconic_content" not in assistant_msg:
                # Find the previous laconic summary to provide sequential context
                previous_laconic_summary = "This is the start of the conversation."
                # Look for the summary on the assistant message of the *previous* turn (i-1)
                if i > 1 and (i - 1) < len(messages):
                    prev_assistant_msg = messages[i-1]
                    previous_laconic_summary = prev_assistant_msg.get("laconic_content", "")

                laconic_prompt = template.format(
                    previous_laconic_summary=previous_laconic_summary,
                    user_content=user_msg['content'],
                    assistant_content=assistant_msg['content']
                )
                summary_response = local_summarizer.generate(laconic_prompt)
                laconic_text = summary_response.get('text', '').strip()

                if laconic_text:
                    assistant_msg["laconic_content"] = laconic_text
                    updated = True
        return updated