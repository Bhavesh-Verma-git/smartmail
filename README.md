# AI Internship Outreach Assistant

A beginner-friendly, production-minded Python tool that helps you send
personalised internship outreach emails using your own profile data and
a language model.

## What this project does

1. Reads your master profile (education, experience, projects, skills).
2. Tailors a CV and an outreach message to each target company.
3. (Future) Sends emails via Gmail API with tracking.

## Project structure

```
smartmail/
+-- data/                  # Your personal data (not committed if real)
|   +-- master_profile.json
+-- src/
|   +-- profile/           # Profile loading and validation
+-- tests/                 # Automated tests
+-- .env.example           # Documents required environment variables
+-- requirements.txt       # Runtime dependencies
+-- requirements-dev.txt   # Dev / test dependencies
```

## Quick start (Windows PowerShell)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Copy the env template and fill in your values
Copy-Item .env.example .env

# 4. Run the tests
pytest tests/ -v
```

## Environment variables

See `.env.example` for the full list. Never commit your real `.env` file.

## Development status

- [x] Step 1 -- Project foundation + master profile structure
- [ ] Step 2 -- Profile tailoring logic
- [ ] Step 3 -- CV generation
- [ ] Step 4 -- Outreach email generation
- [ ] Step 5 -- Gmail integration
