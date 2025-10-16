# Lightweight, self-contained column mapper used in cloud-app
# Provides the same interface as the original IntelligentColumnMapper but uses
# deterministic heuristics (synonyms/normalization) instead of LLMs.

from typing import List, Dict

class IntelligentColumnMapper:
    """
    Drop-in replacement for IntelligentColumnMapper used by ai-app.
    Uses simple normalization + synonyms to map uploaded column names
    to required standard names.
    
    Usage:
        mapper = IntelligentColumnMapper()
        mapping = mapper.map_columns(uploaded_columns, required_columns, "RCM")
    Returns:
        Dict[str, str|None] mapping from required -> original name (or None)
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _normalize(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def map_columns(self, uploaded_columns: List[str], required_columns: List[str], context: str = "") -> Dict[str, str]:
        uploaded_norm = {self._normalize(c): c for c in uploaded_columns}
        # Generic aliases; can be extended per context
        aliases = {
            "entity": ["brandname", "brand", "bu", "businessunit", "division", "entity", "entityname"],
            "controldescription": ["controldesc", "controldescription", "description", "desc", "controldetails"],
            "controlid": ["ctrlid", "controlid", "controlnumber", "id", "control#"],
            "keyyn": ["keycontrol", "key", "iskey", "keyyn", "keyy/n"],
            "accounttype": ["accounttype", "accttype", "type"],
        }
        mapped = {}
        for req in required_columns:
            key = self._normalize(req)
            found = uploaded_norm.get(key)
            if not found and key in aliases:
                for a in aliases[key]:
                    if a in uploaded_norm:
                        found = uploaded_norm[a]
                        break
            if not found:
                for k, v in uploaded_norm.items():
                    if key in k or k in key:
                        found = v
                        break
            mapped[req] = found
        return mapped
