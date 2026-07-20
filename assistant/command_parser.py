"""
Görevi: Kullanıcının komutunu analiz eder. (örn. 'Find my charger' -> intent='find', target='charger')
"""
import re

def parse_command(command: str) -> dict:
    """
    Parses voice or text command to extract intent and target.
    """
    command = command.lower().strip()
    
    # Simple regex extraction
    match = re.search(r"(find|search for|where is|go to|look for)\s+(my\s+|the\s+)?([a-z0-9\s]+)", command)
    
    if match:
        intent = match.group(1)
        target = match.group(3).strip()
        
        # Normalize intents
        if intent in ["find", "search for", "where is", "look for"]:
            intent = "find"
            
        return {"intent": intent, "target": target}
        
    return {"intent": "unknown", "target": None}
