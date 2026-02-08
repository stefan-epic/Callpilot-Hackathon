import os

from dotenv import load_dotenv


load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class Settings:
    twilio_account_sid: str = _require_env("TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = _require_env("TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = _require_env("TWILIO_PHONE_NUMBER")
    elevenlabs_api_key: str = _require_env("ELEVENLABS_API_KEY")
    elevenlabs_agent_id: str = _require_env("ELEVENLABS_AGENT_ID")
    public_base_url: str = _require_env("PUBLIC_BASE_URL").rstrip("/")


settings = Settings()
