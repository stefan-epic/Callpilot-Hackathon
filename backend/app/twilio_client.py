from twilio.rest import Client

from app.config import settings


def get_twilio_client() -> Client:
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)
