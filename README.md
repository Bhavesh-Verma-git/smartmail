# SmartMail: Human-Reviewed AI Internship Outreach Assistant

SmartMail is a local, privacy-first, human-reviewed assistant designed to streamline high-quality outreach for AI/ML engineering internships.

SmartMail automates the time-consuming aspects of company research, technical alignment, and contextual drafting—while keeping **the human user in total control** of every review, edit, and send decision.

---

## Key Features & Design Philosophy

- **Zero Accidental Sends (Dry-Run by Default)**: Live network transmission is disabled by default. Dry-run mode marks actions as `SIMULATED` without making external email API calls.
- **Strict Privacy & Local-First Processing**: Candidate CV bytes, personal names, phone numbers, and contact details remain strictly on your local machine. External LLMs are queried only for public company research unless explicit opt-in is enabled.
- **Evidence-Grounded Personalization**: No hallucinations, fabricated achievements, or fake referral claims. Outreach drafts strictly reference verified technical focus areas from official company websites and corroborated evidence.
- **Uploaded CV vs. Tailored CV Modes**:
  - **Uploaded CV Mode (Default)**: Preserves your original PDF byte-for-byte with SHA-256 integrity verification. Attaches the exact file to your drafts without modification or external LLM parsing.
  - **Tailored CV Mode**: Deterministically selects exactly 3 relevant projects from your structured `master_profile.json` matching the company's verified tech stack and renders a tailored LaTeX PDF.
- **Separation of Classification Concerns**:
  - Email syntax validation & mailbox type (personal vs. generic inboxes like `recruiting@` or `careers@`).
  - Company identity verification & email domain consistency.
  - AI/ML technical relevance based on verifiable products, engineering blogs, and documentation.
  - Internship opportunity status (`APPLICATION_OPEN` vs `APPLICATION_UNKNOWN` vs `APPLICATION_CLOSED`).
- **Resilient Multi-Tier Fallback Engine**:
  - Company Discovery: Tavily Web Search API with automatic fallback to direct HTTP scraping / BeautifulSoup.
  - LLM Inference: Google Gemini API as primary, Groq API as configurable fallback, and deterministic rule-based offline generation if external providers are unavailable.

---

## Architecture & Six Workflow Stages

```
[ Candidate CV / Master Profile ]  +  [ Contact List (CSV / Excel) ]
                                |
                                v
               +----------------------------------+
               | 1. Upload CV                     |
               | - Byte-for-byte PDF preservation |
               | - SHA-256 digest verification    |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               | 2. Import Contacts               |
               | - CSV / XLSX / XLS ingestion     |
               | - Column mapping & syntax checks |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               | 3. Research & Verify             |
               | - Company domain & identity      |
               | - AI/ML relevance assessment     |
               | - Recipient identity & vacancy   |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               | 4. Generate Drafts               |
               | - Truthful personalization       |
               | - Evidence & profile grounding   |
               | - Exact CV PDF attachment        |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               | 5. Review & Approve              |
               | - Inspect research evidence      |
               | - Edit subject / body            |
               | - Approval invalidation on edits |
               | - Approve, Reject, or Hold       |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               | 6. Dry-Run & Audit               |
               | - Simulated delivery recording   |
               | - Duplicate-send prevention lock |
               | - SQLite audit trail persistence |
               +----------------------------------+
```

---

## Installation & Setup (Windows PowerShell)

### 1. Prerequisites
- **Python 3.11+** installed and available in PATH.
- (Optional) LaTeX distribution (MiKTeX or TeX Live) if compiling PDFs in *Tailored CV Mode*.

### 2. Virtual Environment Setup
```powershell
# Navigate to the repository root
cd C:\Users\verma\Desktop\smartmail

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Environment Configuration
Copy the template `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Edit `.env` to configure your settings and optional API keys:
```ini
# Security & Flask
FLASK_SECRET_KEY=generate_or_set_secure_random_hex
DRY_RUN=true
SMARTMAIL_SHARE_CV_WITH_EXTERNAL_LLM=false

# Search & LLM Providers (Free Tiers Supported)
TAVILY_API_KEY=tvly-your-optional-tavily-key
GEMINI_API_KEY=AIzaSy-your-gemini-key
GEMINI_MODEL=gemini-3.8-flash
GROQ_API_KEY=gsk_your-groq-key
GROQ_MODEL=openai/gpt-oss-120b
```

Check your configuration status anytime via CLI (keys are never printed or leaked):
```powershell
python -m src.cli config-status
```

---

## Running the Application

### Web Review Interface (Flask UI)
Start the local review interface:
```powershell
python -m src.ui.app
```
Then open `http://127.0.0.1:5000` in your web browser.

**The Six Workflow Stages:**
1. **Upload CV**: Register standard resume PDF (stored securely in output/uploaded_cvs/ with SHA-256 checksum).
2. **Import Contacts**: Import CSV/XLS/XLSX files, map columns, validate rows, and review pre-flight errors before confirmation.
3. **Research & Verify**: Perform company discovery, domain verification, recipient name verification, and AI/ML technical relevance assessment.
4. **Generate Drafts**: Synthesize truthful personalized emails grounded in verified company evidence and candidate profile facts.
5. **Review & Approve**: Human-in-the-loop draft inspection, body/subject editing (with approval invalidation), approval, and rejection/hold.
6. **Dry-Run & Audit**: Simulate dispatch without network transmission and inspect persistent audit records in SQLite (data/send_log.db).

### CLI Interface
SmartMail can also be managed via the CLI:
```powershell
# View available CLI commands
python -m src.cli --help

# Check provider configuration
python -m src.cli config-status
```

---

## Testing & Verification

SmartMail has an extensive test suite covering unit operations, security safeguards, mock providers, parsing, and pipeline integration.

```powershell
# Run the complete test suite
.\.venv\bin\python.exe -m pytest

# Run tests with short progress report
.\.venv\bin\python.exe -m pytest -q

# Run specific test suites
.\.venv\bin\python.exe -m pytest tests/test_email_safeguards.py
.\.venv\bin\python.exe -m pytest tests/test_stage3_research_classification.py
.\.venv\bin\python.exe -m pytest tests/test_ui.py
```

### Mocked vs. Live Tests
- **Automated Test Suite**: All tests in `tests/` use deterministic fixtures, simulated HTTP responses, and mocked provider interfaces. No real emails are sent and no external API tokens are consumed during `pytest` runs.
- **Provider Smoke Testing**: Provider integration tests verify credentials safely and ensure structured output schemas without exposing keys or personal data.

---

## Security, Privacy & Safeguards

1. **Email Sending Safeguards**:
   - `EmailSender.send()` enforces `dry_run` verification *before* calling any provider.
   - Live network transmission requires both `DRY_RUN=false` and an explicit live provider configuration.
   - Draft fields (`recipient`, `subject`, `body`, `attachment_path`) are strictly validated prior to dispatch.
   - Editing a draft immediately resets any prior `APPROVED` status back to `DRAFT`.
   - Concurrency-safe duplicate prevention and atomic reservations are tracked in SQLite (`data/send_log.db`) across server restarts and worker threads.
2. **Flask Security**:
   - CSRF protection enforced across all state-changing forms and API endpoints with session-bound tokens.
   - Maximum request payload limit enforced (16 MB) with chunked streaming CV file validation preventing unbounded memory usage.
   - Session cookies configured with `HttpOnly` and `SameSite=Lax` (and `Secure` in production).
   - Random secret key generation in development with strict validation in production mode (`FLASK_SECRET_KEY`).
   - Path traversal prevention on file token routes, draft review IDs, and CV download endpoints (`.pdf` restriction to allowed directories).
3. **Data Protection**:
   - `.env`, `credentials.json`, `token.json`, uploaded CVs, private draft JSON, session state, and SQLite databases are excluded by `.gitignore`.
   - Contact datasets included in the repository contain only RFC 2606 synthetic `.example` domains.

---

## Known Limitations & Future Work

- **Tailored CV LaTeX Engine**: Compiling tailored PDFs in *Tailored CV Mode* requires a local TeX environment (`pdflatex` or `xelatex`). If LaTeX is not installed, SmartMail seamlessly operates in *Uploaded CV Mode*.
- **Future Live Sending**: Direct Gmail API sending is architected as an optional future provider. Live sending remains intentionally disabled in this release to prioritize verification accuracy and user review safety.
