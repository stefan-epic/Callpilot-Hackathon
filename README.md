# Callpilot Hackathon

## Project Summary
Callpilot is an AI receptionist that turns a single Telegram message into a booked appointment. You text what you need, and the system orchestrates the entire workflow: n8n extracts the details, Twilio places the call, and ElevenLabs handles the conversation with the business in real time. Calendar constraints are enforced automatically to avoid double booking, and the full transcript is sent back to n8n for analysis. With n8n, we build an easily extendable workflow that allows for easy implementation of additional features. At the end of the workflow, n8n will automatically create the calendar entry and get back to you on Telegram. Using the ElevenLabs Conversational SDK, the AI can understand and respond blazingly fast, in under 1 second. The result is a hands-off scheduling experience that feels like having a dedicated assistant. Call smart, call pilot.

## How to Run
### Accounts Required
- Twilio account
- ElevenLabs account
- n8n account

### Configure Environment
1. Copy the example env file into an actual env file and fill it out:
   - `backend/app/.env.example`
2. Provide:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `TWILIO_PHONE_NUMBER`
   - `ELEVENLABS_API_KEY`
   - `ELEVENLABS_AGENT_ID`
   - `PUBLIC_BASE_URL`

### Start the Backend
```
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Configure n8n Workflows
1. In CallPilot.json, replace the placeholder `{PHONE_NUMBER}` **twice** with the real number you want to call.
2. Import both workflow files into n8n:
   - `CallPilot.json`
   - `WebHookCallPilot.json`
3. Ensure the callback URL points to your webhook endpoint in n8n.

