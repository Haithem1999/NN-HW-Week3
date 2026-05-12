# AI Voice Agent Templates — Auto Repair Shops

Supplementary materials accompanying the thesis *"Enhancing Operational Efficiency in U.S. Automotive Repair Shops"* by Gasmi Haithem Aissa Khalil.

This folder contains generic, reusable templates of the voice agent and the supporting workflow automations that were adapted and deployed across the three independent auto repair shops studied in the thesis (Engines Express, John's Automotive Care, Miramar Automotive). Shop-specific configuration values, contact details, scheduling URLs, and proprietary business information have been replaced with bracketed placeholders (e.g. `[AUTO REPAIR SHOP]`, `[SCHEDULING LINK]`, `[TOW TRUCK PHONE NUMBER]`) so the templates can be reused without exposing any of the partner shops' private data.

## Contents

### `retell-ai-voice-agent/`
A Retell AI voice agent blueprint (`auto-repair-after-hours.json`) covering the after-hours scenario. The blueprint defines the multi-state finite state machine described in Chapter 4 of the thesis, including:

- General prompt and identity configuration
- Seven conversational states: `main`, `appointment_link`, `appointment_request`, `quote`, `message`, `urgent_request`, `towing`
- Function-calling tool definitions for each transactional state (`appointment_link`, `quote`, `message`, `contactTow`, `appointment_request`, `urgent_message`)
- Voice settings (voice ID, temperature, speed, volume, interruption sensitivity, responsiveness)
- Post-call analysis schema (summary prompt and `appointment_activity` boolean)

Import this JSON directly into Retell AI to recreate the agent, then replace the bracketed placeholders with shop-specific values.

### `make-workflow/`
A Make.com scenario blueprint (`auto-repair-ai-functions.blueprint.json`) that handles the downstream automation triggered by each of the voice agent's function calls. The scenario uses a single inbound webhook with a basic router that branches on the function name (`message`, `contactTow`, `quote`, `appointment_link`, `appointment_request`, `urgent_message`) and dispatches:

- **SendGrid** transactional email sends (to the shop) using dynamic templates
- **Twilio** SMS sends (to the caller, and where relevant to the tow company)
- Webhook responses back to the voice agent

Import the blueprint into Make.com, reconnect your own SendGrid and Twilio accounts, point the SendGrid template IDs at your own dynamic templates, and update the destination email addresses and phone numbers.

### `n8n-workflow/`
An n8n workflow (`post-call-records.json`) that ingests Retell AI's `call_analyzed` webhook events and persists the structured call record (transcript, summary, cost, latency, sentiment, disconnection reason, duration, recording URL, etc.) into a Supabase `all_client_calls` table. The workflow filters out non-phone-call events and skips records that contain tool-call invocations (these are handled by the Make scenario above to avoid duplicate billing/notifications). This is the same analytics pipeline referenced in Chapter 4 §4.3.5 and Chapter 5 of the thesis.

### `analysis-scripts/`
The Python scripts used to produce the quantitative and qualitative findings in Chapter 5:

| Script | Purpose |
| --- | --- |
| `01_extract_transcripts.py` | Parses the per-shop post-call CSV exports, structures each transcript into turn-by-turn JSON, detects caller intents, sentiment markers, AI-disclosure, and human-request flags. |
| `02_thematic_analysis.py` | Applies the deductive-inductive coding framework (Appendix A of the thesis) — trust, helpfulness, empathy, and emergent codes — across all transcripts and emits aggregate frequencies and per-call code counts. |
| `03_statistics_summary.py` | Consolidates the outputs of Scripts 1 and 2 into the descriptive statistics, conversation-quality metrics, intent distributions, tool utilisation, cross-tabulations, temporal trends, and thesis-ready CSV tables presented in Chapter 5. |

The scripts use only the Python standard library. Update `DOWNLOADS` and `OUTPUT_DIR` at the top of each script to match your environment before running.

## Reuse and Adaptation

To deploy these templates for a new repair shop:

1. Import the Retell AI blueprint and fill in shop-specific FAQs, services, hours, address, and the bracketed placeholders in each state prompt.
2. Import the Make.com scenario, reconnect SendGrid/Twilio, and substitute the recipient email, the tow company phone number, the scheduling link, and the SendGrid dynamic template IDs.
3. Import the n8n workflow, reconnect Supabase, and update the `company_id` field to identify the shop in the shared analytics table.
4. Point the voice agent's tool webhook URLs at the imported Make scenario, and point Retell AI's post-call webhook at the imported n8n workflow.

## Note on Bracketed Placeholders

All deployment-specific values appear as `[BRACKETED_LABELS]` in the templates. Replace them before deploying:

- `[AUTO REPAIR SHOP]` / `[AUTO REPIAR SHOP]` — the shop's display name
- `[SHOP FULL ADDRESS]` — the shop's street address
- `[SHOP HOURS OF OPERATION]` — opening hours per day
- `[SCHEDULING LINK]` — the booking/scheduling URL sent via SMS
- `[TOW TRUCK COMPANY]` / `[TOW TRUCK PHONE NUMBER]` — preferred tow partner contact
- `[INSERT IANA Time Zone Identifier]` — e.g. `America/Los_Angeles`

Sender phone numbers, webhook hook IDs, and connection IDs inside the Make/n8n blueprints are project-specific and will be re-bound automatically when you import the blueprints into your own workspaces.
