import json
from pathlib import Path
from collections import defaultdict
import re

class SkillManager:
    """Discovers, loads, and manages agentic skills from a repository."""
    def __init__(self, skills_base_dir: str = "antigravity_skills"):
        self.base_dir = Path(skills_base_dir)
        self.skills_index_path = self.base_dir / "skills_index.json"
        self.skills_dir = self.base_dir / "skills"
        self.skills_data = defaultdict(dict)
        self.skills_found = False
        self._load_skills()

    def _load_skills(self):
        """Loads skills from the flat list in skills_index.json."""
        if not self.skills_index_path.exists():
            return

        try:
            # index_data is a LIST of dictionaries
            index_data = json.loads(self.skills_index_path.read_text(encoding="utf-8"))
            
            for skill_info in index_data:
                # Extract the category to group them, defaulting to 'uncategorized'
                category = skill_info.get("category", "uncategorized")
                skill_id = skill_info.get("id")
                
                if skill_id:
                    self.skills_data[category][skill_id] = {
                        "name": skill_info.get("name", skill_id),
                        "description": skill_info.get("description", ""),
                        "path": self.base_dir / skill_info.get("path", "")
                    }
            
            self.skills_found = True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading skills index: {e}")
            self.skills_found = False

    def get_all_skills(self) -> dict:
        """Returns the categorized dictionary of all loaded skills."""
        return self.skills_data

    def get_skill_content(self, skill_display_name: str) -> str | None:
        """
        Retrieves the content of a skill file using its display name.
        The display name is in the format "[Category] Skill Name".
        """
        # Extract category and skill name from the display format
        match = re.match(r'^\[(.*?)\]\s*(.*)$', skill_display_name)
        if not match:
            return None
        
        category, skill_name_from_display = match.groups()

        if category in self.skills_data:
            for skill_id, skill_info in self.skills_data[category].items():
                if skill_info['name'] == skill_name_from_display:
                    skill_path = skill_info.get('path')
                    if skill_path and Path(skill_path).exists():
                        try:
                            return Path(skill_path).read_text(encoding="utf-8")
                        except IOError as e:
                            print(f"Error reading skill file {skill_path}: {e}")
                            return None
        return None