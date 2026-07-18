import logging
import json
from datetime import datetime

def log_agent_interaction(query, response, latency, model_version):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "latency_ms": latency,
        "model": model_version
    }
    # In production, write this to Azure Monitor or a central DB
    print(f"AUDIT LOG: {json.dumps(log_entry)}")