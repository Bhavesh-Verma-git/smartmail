import json
import sys
from pathlib import Path
sys.path.insert(0, ".")

from src.ui.storage import regenerate_draft_content, get_draft

batch_path = Path("output/.batches/batch_af3a5b94e2.json")
if not batch_path.exists():
    print("Batch file not found.")
    sys.exit(1)

batch_data = json.loads(batch_path.read_text(encoding="utf-8"))
results = batch_data.get("results", [])

print(f"Refreshing {len(results)} drafts in batch_af3a5b94e2:\n")

for r in results:
    draft_id = r.get("draft_id")
    company = r.get("company_name")
    if not draft_id:
        continue
    try:
        updated = regenerate_draft_content(draft_id)
        pr = updated.get("personalization_record", {})
        finding = pr.get("company_finding", "EMPTY")
        domain = pr.get("technical_domain", "UNKNOWN")
        print(f"UPDATED: {draft_id} | {company}")
        print(f"  DOMAIN:  {domain}")
        print(f"  FINDING: {finding[:90]}...")
        print(f"  STATUS:  {updated.get('generation_status')}")
        print()
    except Exception as exc:
        print(f"ERROR updating {draft_id} ({company}): {exc}")

print("Batch 209 drafts successfully refreshed.")
