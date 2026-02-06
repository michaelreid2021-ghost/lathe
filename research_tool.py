import requests
import os
import re
import json
from urllib.parse import urlparse, unquote
from pathlib import Path
from bs4 import BeautifulSoup
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_models import LocalModel

UA_DESKTOP = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

class ResearchManager:
    """Manages fetching, storing, and providing context from web URLs."""

    def __init__(self, session_dir: Path):
        self.research_dir = session_dir / "research_sources"
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.research_dir / "_summary.json"
        self.sources = {}  # url -> {"filename": str, "summary": str, "status": str, "enabled": bool}
        self._load_state()

    def _load_state(self):
        """Loads the state of research sources from a JSON file."""
        if self.summary_path.exists():
            try:
                self.sources = json.loads(self.summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                self.sources = {} # Start fresh if file is corrupt

    def _save_state(self):
        """Saves the current state of research sources to a JSON file."""
        self.summary_path.write_text(json.dumps(self.sources, indent=2), encoding="utf-8")

    def _sanitize_url(self, raw_url: str) -> str:
        """Refangs and cleans URLs."""
        url = unquote(raw_url.strip())
        url = url.replace('[.]', '.').replace('hxxp', 'http').replace('[:]', ':').replace('[/]', '/')
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        return url

    def _clean_filename(self, url: str) -> str:
        """Converts a URL to a safe filename."""
        s = re.sub(r'(^\w+:|^)\/\/', '', url)
        return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', s)[:100]

    def _process_and_save_content(self, content: str, url: str) -> tuple[bool, str, int, str]:
        """Parses content, strips unwanted elements, and saves to disk."""
        clean_name = self._clean_filename(url)
        filepath = self.research_dir / f"{clean_name}.txt"
        
        cleaned_text = ""
        try:
            # Use BeautifulSoup to clean HTML, but handle plain text gracefully
            if bool(BeautifulSoup(content, "html.parser").find()):
                soup = BeautifulSoup(content, 'html.parser')
                for element in soup(["script", "style", "head", "nav", "footer", "aside"]):
                    element.decompose()
                # Get text and clean up whitespace
                cleaned_text = ' '.join(soup.get_text().split())
            else:
                # It's likely plain text
                cleaned_text = content

            filepath.write_text(cleaned_text, encoding='utf-8', errors='ignore')
            file_size = filepath.stat().st_size
            return True, str(filepath), file_size, cleaned_text
        except Exception:
            return False, "", 0, ""

    def add_url(self, raw_url: str, local_model: 'LocalModel', summary_template: str) -> tuple[bool, str]:
        """Fetches a URL, processes it, summarizes it, and adds it to the managed sources."""
        target_url = self._sanitize_url(raw_url)
        if target_url in self.sources:
            return False, "This URL has already been added."

        headers = {"User-Agent": UA_DESKTOP}
        try:
            # Disable SSL warnings for this operation, as in the original script
            requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.get(
                target_url,
                headers=headers,
                timeout=20,
                verify=False,
                allow_redirects=True
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            success, filepath, size, cleaned_text = self._process_and_save_content(response.text, target_url)

            if not success or not cleaned_text.strip():
                self.sources[target_url] = {"filename": "", "summary": "", "status": "Failed: No content", "enabled": False}
                self._save_state()
                return False, "Processing failed: The cleaned content was empty."

            # Generate summary
            summary_prompt = summary_template.format(filename=target_url, content=cleaned_text[:10000]) # Limit summary content
            summary = local_model.generate(summary_prompt).get("text", "Error summarizing content.")

            self.sources[target_url] = {
                "filename": Path(filepath).name,
                "summary": summary,
                "status": f"Success ({size:,} bytes)",
                "enabled": True
            }
            self._save_state()
            return True, f"Successfully captured and summarized {target_url}."

        except requests.exceptions.RequestException as e:
            self.sources[target_url] = {"filename": "", "summary": "", "status": f"Failed: {e}", "enabled": False}
            self._save_state()
            return False, f"Failed to fetch URL: {e}"
        except Exception as e:
            return False, f"An unexpected error occurred: {e}"

    def toggle_source(self, url: str, enabled: bool):
        """Enables or disables a source for context inclusion."""
        if url in self.sources:
            self.sources[url]["enabled"] = enabled
            self._save_state()

    def remove_source(self, url: str):
        """Removes a source and its associated file."""
        if url in self.sources:
            source_data = self.sources.pop(url)
            filename = source_data.get("filename")
            if filename:
                filepath = self.research_dir / filename
                filepath.unlink(missing_ok=True)
            self._save_state()

    def get_enabled_context(self) -> str:
        """Constructs a context string from all enabled research sources."""
        context_parts = []
        for url, data in self.sources.items():
            if data.get("enabled") and data.get("filename"):
                filepath = self.research_dir / data["filename"]
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    context_parts.append(f"--- RESEARCH SOURCE: {url} ---\n{content}\n--- END SOURCE: {url} ---")
        
        if not context_parts:
            return ""
            
        return "\n\n" + "\n\n".join(context_parts)
