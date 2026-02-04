import re
import time
import datetime
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from session_manager import SessionContext

class PromptBuilder:
    """Orchestrates the construction of the final prompt for the AI."""
    def __init__(self,
                 session_context: 'SessionContext',
                 artifact_context_levels: dict,
                 provisional_context_enabled: bool,
                 provisional_context_text: str,
                 last_interaction_time: float,
                 format_timedelta: Callable[[float], str],
                 staged_revisions: dict = None):
        self.session_context = session_context
        self.artifact_context_levels = artifact_context_levels
        self.provisional_context_enabled = provisional_context_enabled
        self.provisional_context_text = provisional_context_text
        self.last_interaction_time = last_interaction_time
        self.format_timedelta = format_timedelta
        self.staged_revisions = staged_revisions if staged_revisions is not None else {}

    def build_prompt(self, user_input: str) -> tuple[str, str]:
        """
        Constructs the full AI prompt and a user-facing display text.

        Returns:
            A tuple containing:
            - The complete prompt string to be sent to the AI model.
            - The formatted text to display in the chat UI for the user's message.
        """
        temporal_context = self._get_temporal_context()
        ai_inference_prompt, explicitly_referenced_files, ref_count = self._resolve_references(user_input)
        
        general_context = self._get_general_context(explicitly_referenced_files)
        provisional_context = self._get_provisional_context()

        full_prompt_to_ai = (
            temporal_context +
            ai_inference_prompt +
            "\n--- CURRENT GENERAL CONTEXT ---\n" +
            general_context +
            "\n" + provisional_context
        ).strip()

        display_text = user_input
        if ref_count > 0:
            display_text = f"📎 *[Referenced {ref_count} item(s)]* " + user_input

        return full_prompt_to_ai, display_text

    def _get_temporal_context(self) -> str:
        """Generates the temporal context string."""
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_utc_iso = now_utc.isoformat()

        return (
            f"current time: {now_utc_iso}\n"
            f"time since last interaction: {self.format_timedelta(time_since_last)}.\n"
        )

    def _resolve_references(self, text: str) -> tuple[str, list[str], int]:
        """Finds and replaces XML-style references with their content."""
        ref_pattern = r'<(\w+) name="([^"]+)"/>'
        matches = re.findall(ref_pattern, text)

        explicitly_referenced_files = []
        ai_inference_prompt = text
        ref_count = len(matches)

        for full_tag_text, tag_type, name in [(f'<{m[0]} name="{m[1]}"/>', m[0], m[1]) for m in matches]:
            if name in self.session_context.artifacts:
                explicitly_referenced_files.append(name)
                stable_data = self.session_context.artifacts[name]
                is_staged = name in self.staged_revisions

                content_payload = f"# Could not resolve reference for {name}"
                source_path_str = stable_data.get("source_path")
                source_path = Path(source_path_str) if source_path_str else None

                # If the file is staged, any explicit reference uses the LIVE source code from its source_path
                # and is promoted to a FOCAL reference to draw the AI's attention to the change.
                if is_staged and source_path and source_path.exists():
                    raw_code = source_path.read_text(encoding="utf-8")
                    content_payload = f"```\n{raw_code}\n```"
                    tag_type = "focal"
                # If not staged, handle based on tag type using the session's stable working copies.
                else:
                    if tag_type == "focal":
                        # Use the stable working copy for focal, consistent with get_context_string
                        working_path = stable_data.get('path')
                        if working_path and working_path.exists():
                            raw_code = working_path.read_text(encoding="utf-8")
                            content_payload = f"```\n{raw_code}\n```"
                    elif tag_type == "dependency":
                        skeleton_content = '# Skeleton unavailable'
                        skeleton_path = stable_data.get('skeleton_path')
                        if skeleton_path and Path(skeleton_path).exists():
                            skeleton_content = Path(skeleton_path).read_text(encoding="utf-8")
                        content_payload = f"```python\n{skeleton_content}\n```"
                    else: # background or other
                        summary = stable_data.get('summary', '# Summary unavailable')
                        content_payload = f"\n{summary}\n"

                expanded_block = f'--- {tag_type.upper()} CONTEXT: {name} ---\n{content_payload}\n--- END {tag_type.upper()} CONTEXT: {name} ---'
                ai_inference_prompt = ai_inference_prompt.replace(full_tag_text, expanded_block)

        return ai_inference_prompt, explicitly_referenced_files, ref_count

    def _get_general_context(self, explicitly_referenced_files: list[str]) -> str:
        """Gets the context string for all non-explicitly referenced artifacts."""
        return self.session_context.get_context_string(
            self.artifact_context_levels,
            manually_inserted_files=explicitly_referenced_files,
            staged_revisions=self.staged_revisions
        )

    def _get_provisional_context(self) -> str:
        """Returns the provisional context text if it's enabled."""
        if self.provisional_context_enabled and self.provisional_context_text.strip():
            return f"--- SCRATCH PAD / PROVISIONAL CONTEXT ---\n{self.provisional_context_text}"
        return ""