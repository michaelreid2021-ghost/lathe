import os
import json
import requests
import re
import random
from bs4 import BeautifulSoup

# --- Configuration ---
BASE_URL = os.environ.get("LUCIDUM_BASE_URL", "https://hackathon.lucidum.cloud/CMDB")
AUTH_TOKEN = os.environ.get("LUCIDUM_AUTH", "Bearer VjUpJwQFNODgrCVoVtyM") 
def sess():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json", "Authorization": AUTH_TOKEN})
    s.base_url = BASE_URL
    return s

def query_ldg(s, table: str, query: list, limit: int = 50):
    url = f'/v2/data/ldg'
    full_url = s.base_url + url
    body = {"query": query, "table": table, "paging": {"page": 0, "recordsPerPage": limit}}
    try:
        r = s.post(full_url, json=body, timeout=15)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        return []

# --- 1. OSINT SCRAPER ---
def osint_threat_lookup(ioc: str = "", type: str = "mixed") -> str:
    print(f"--- TOOL: OSINT Lookup Input: {ioc[:50]}... ---")
    
    # Extract URL from input string
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*')
    found_urls = url_pattern.findall(ioc)
    original_url = found_urls[0] if found_urls else ioc
    
    # Sanitize for Requests (Requests hates fragments #)
    request_url = original_url.split('#')[0]
    
    if request_url.startswith("http"):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            
            response = requests.get(request_url, headers=headers, timeout=15, verify=False)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Strip junk
                for element in soup(["script", "head", "style", "nav", "footer", "iframe", "noscript"]):
                    element.decompose()
                
                # Clean text
                clean_text = ' '.join(soup.get_text(separator=' ').split())[:12000]
                title = soup.title.string if soup.title else "No Title"
                
                return json.dumps({
                    "status": "Success", 
                    "original_url": original_url, # Key for LLM to find fragment CVEs
                    "title": title,
                    "content_preview": clean_text 
                })
            else:
                return json.dumps({"status": "Error", "message": f"HTTP {response.status_code}", "url": original_url})
                
        except Exception as e:
            return json.dumps({"status": "Error", "message": str(e), "url": original_url})
            
    return json.dumps({"status": "No_URL_Found", "ioc": ioc})

# --- 2. LUCIDUM SEARCH ---
def find_vulnerable_assets(cve_id: str):
    # Sanitize
    clean_cve = str(cve_id).strip().strip('"').strip("'").strip().upper()
    
    if not clean_cve or "NONE" in clean_cve or len(clean_cve) < 5:
         return json.dumps({"status": "Skipped", "message": "No valid CVE ID provided."})

    try:
        s = sess()
        
        # CORRECT FIELD: Vuln_List.CVE
        filters = [[{
            "searchFieldName": "Vuln_List.CVE", 
            "operator": "==", 
            "type": "String", 
            "value": clean_cve
        }]]

        rows = query_ldg(s, "asset", filters, limit=50)
        
        if not rows:
            return json.dumps({
                "status": "Clean", 
                "count": 0, 
                "message": f"Lucidum returned 0 assets for {clean_cve}."
            })
        
        results = []
        for row in rows:
            results.append({
                "hostname": row.get("Asset_Name") or row.get("Latest_Asset_Name"),
                "ip": (row.get("IP_Address") or row.get("EXT_IP_Address") or ["N/A"])[0],
                "os": row.get("OS") or row.get("Lucidum_OS"),
                "criticality": "HIGH"
            })
        
        return json.dumps({"status": "VULNERABLE", "count": len(results), "assets": results})

    except Exception as e:
        return json.dumps({"error": f"Lucidum API Error: {str(e)}"})

# --- MOCKS ---
def siem_query_executor(generated_query_string: str) -> str:
    return json.dumps({"status": "Query Executed", "hits": 0, "message": "No events found matching the query."})

def ticket_create_itsm(summary: str, queue: str, priority: str, description: str = "") -> str:
    return json.dumps({"ticket_id": f"INC-{random.randint(10000, 99999)}", "status": "Created"})

def apply_security_rule(indicator: str, comment: str) -> str:
    return json.dumps({"status": "Success", "rule_id": "BLOCK-99"})

TOOL_MAP = {
    "OSINT_Threat_Lookup": osint_threat_lookup,
    "SIEM_Query_Executor": siem_query_executor,
    "Ticket_Create_ITSM": ticket_create_itsm,
    "Apply_Security_Rule": apply_security_rule,
    "Lucidum_Enrichment": lambda records: json.dumps({"enriched": True}), 
    "Lucidum_CVE_Search": find_vulnerable_assets
}