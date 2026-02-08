import asyncio
import base64
import json
import logging
import re
import time
from typing import Any

import audioop
import httpx
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.config import settings
from app.twilio_client import get_twilio_client


logger = logging.getLogger("callpilot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI()


class CallRequest(BaseModel):
    to: str = Field(..., description="E.164 formatted phone number")
    context: str | None = Field(
        default=None,
        description="Optional contextual data to send to ElevenLabs at call start",
    )
    calendar: Any | None = Field(
        default=None,
        description="Optional Google Calendar payload to derive blocked ranges",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional n8n webhook URL to notify when call completes",
    )


# In-memory context store (call_sid -> context text).
# This keeps the prototype simple and avoids any database.
CALL_CONTEXT: dict[str, str] = {}
# In-memory callback store (call_sid -> webhook URL).
CALL_CALLBACKS: dict[str, str] = {}
# In-memory transcript store (call_sid -> list of lines).
CALL_TRANSCRIPTS: dict[str, list[str]] = {}


def _extract_event_ranges(payload: Any) -> list[str]:
    """
    Extract start/end date values using regex over the raw JSON payload.
    This allows us to accept full Google Calendar objects and filter to dates only.
    """
    raw = json.dumps(payload, ensure_ascii=False)
    start_matches = re.findall(
        r'"start"\s*:\s*\{[^}]*?"(dateTime|date)"\s*:\s*"([^"]+)"', raw
    )
    end_matches = re.findall(
        r'"end"\s*:\s*\{[^}]*?"(dateTime|date)"\s*:\s*"([^"]+)"', raw
    )
    starts = [match[1] for match in start_matches]
    ends = [match[1] for match in end_matches]
    ranges: list[str] = []
    for start_value, end_value in zip(starts, ends):
        ranges.append(f"{start_value} to {end_value}")
    return ranges


async def _get_elevenlabs_signed_url() -> str:
    url = "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url"
    headers = {"xi-api-key": settings.elevenlabs_api_key}
    params = {"agent_id": settings.elevenlabs_agent_id}

    async with httpx.AsyncClient(timeout=10.0) as client:
        logger.info("Requesting ElevenLabs signed URL (agent_id=%s)", params["agent_id"])
        response = await client.get(url, headers=headers, params=params)
        logger.info(
            "ElevenLabs signed URL response status=%s",
            response.status_code,
        )
        response.raise_for_status()
        data = response.json()

    signed_url = data.get("signed_url")
    if not signed_url:
        logger.error("ElevenLabs response missing signed_url: %s", data)
        raise RuntimeError("ElevenLabs signed_url missing in response")
    logger.info("ElevenLabs signed URL acquired")
    return signed_url


def _public_ws_base_url() -> str:
    if settings.public_base_url.startswith("https://"):
        return "wss://" + settings.public_base_url[len("https://") :]
    if settings.public_base_url.startswith("http://"):
        return "ws://" + settings.public_base_url[len("http://") :]
    return settings.public_base_url


def _to_twilio_ulaw_8000(
    audio_base64: str,
    *,
    audio_format: str | None,
    sample_rate: int | None,
) -> str:
    """
    Convert ElevenLabs audio to Twilio-compatible mu-law 8k.
    Falls back to passthrough if format is already ulaw_8000.
    """
    if audio_format and audio_format.lower() in {"ulaw_8000", "mulaw_8000", "mulaw"}:
        return audio_base64

    raw_audio = base64.b64decode(audio_base64)

    # Handle raw PCM 16-bit little endian as the most common case.
    pcm = raw_audio
    in_rate = sample_rate or 16000
    if in_rate != 8000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, in_rate, 8000, None)

    ulaw = audioop.lin2ulaw(pcm, 2)
    return base64.b64encode(ulaw).decode("ascii")


def _twilio_ulaw_to_pcm16_16k(audio_base64: str) -> str:
    raw_ulaw = base64.b64decode(audio_base64)
    pcm16_8k = audioop.ulaw2lin(raw_ulaw, 2)
    pcm16_16k, _ = audioop.ratecv(pcm16_8k, 2, 1, 8000, 16000, None)
    return base64.b64encode(pcm16_16k).decode("ascii")


@app.post("/call")
async def place_call(payload: CallRequest) -> dict:
    # Twilio outbound call is triggered here.
    try:
        logger.info(
            "Placing outbound call to=%s from=%s voice_url=%s",
            payload.to,
            settings.twilio_phone_number,
            f"{settings.public_base_url}/twilio/voice",
        )
        started_at = time.monotonic()
        client = get_twilio_client()
        call = client.calls.create(
            to=payload.to,
            from_=settings.twilio_phone_number,
            url=f"{settings.public_base_url}/twilio/voice",
            method="POST",
            status_callback=f"{settings.public_base_url}/twilio/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST",
        )
        context_parts: list[str] = []
        if payload.context:
            context_parts.append(payload.context)
        if payload.calendar:
            ranges = _extract_event_ranges(payload.calendar)
            if ranges:
                context_parts.append("Blocked appointment times: " + ", ".join(ranges))
        if context_parts:
            CALL_CONTEXT[call.sid] = " ".join(context_parts)
            logger.info("Stored call context for sid=%s", call.sid)
        if payload.callback_url:
            CALL_CALLBACKS[call.sid] = payload.callback_url
            logger.info(
                "Stored callback URL for sid=%s url=%s",
                call.sid,
                payload.callback_url,
            )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.info("Outbound call initiated sid=%s elapsed_ms=%s", call.sid, elapsed_ms)
        return {"status": "initiated", "call_sid": call.sid}
    except Exception as exc:
        logger.exception("Failed to initiate outbound call: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to initiate call")


@app.post("/twilio/voice")
async def twilio_voice() -> Response:
    # ElevenLabs Conversational AI is connected here via Twilio <Stream>.
    try:
        logger.info("Twilio webhook received, building TwiML")
        voice_response = VoiceResponse()
        connect = Connect()
        connect.stream(url=f"{_public_ws_base_url()}/twilio/stream")
        voice_response.append(connect)
        twiml = str(voice_response)
        logger.info("TwiML generated, length=%s", len(twiml))
        return Response(content=twiml, media_type="text/xml")
    except Exception as exc:
        logger.exception("Failed to build TwiML: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build TwiML")


@app.post("/twilio/status")
async def twilio_status(request: Request) -> dict:
    """
    Twilio status callback: when call completes, notify n8n (if provided).
    """
    form = await request.form()
    payload = dict(form)
    call_sid = payload.get("CallSid")
    call_status = payload.get("CallStatus")
    logger.info("Twilio status callback sid=%s status=%s", call_sid, call_status)
    if call_sid and call_status == "completed" and call_sid in CALL_CALLBACKS:
        callback_url = CALL_CALLBACKS.pop(call_sid)
        transcript_lines = CALL_TRANSCRIPTS.pop(call_sid, [])
        notify_payload = {
            "call_sid": call_sid,
            "status": call_status,
            "from": payload.get("From"),
            "to": payload.get("To"),
            "duration": payload.get("CallDuration"),
            "transcript": "\n".join(transcript_lines),
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(callback_url, json=notify_payload)
            logger.info("Notified n8n callback for sid=%s", call_sid)
        except Exception as exc:
            logger.exception("Failed to notify n8n callback: %s", exc)
    return {"status": "ok"}


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Twilio media stream connected")

    try:
        signed_url = await _get_elevenlabs_signed_url()
        async with websockets.connect(signed_url, ping_interval=None) as eleven_ws:
            await eleven_ws.send(
                json.dumps(
                    {
                        "type": "conversation_initiation_client_data",
                        "conversation_initiation_client_data": {
                            "input_audio_format": "ulaw_8000",
                            "output_audio_format": "ulaw_8000",
                        },
                    }
                )
            )
            stream_sid: str | None = None
            current_call_sid: str | None = None
            input_format = "ulaw_8000"
            output_format = "ulaw_8000"

            async def forward_twilio_to_eleven() -> None:
                nonlocal stream_sid, input_format, current_call_sid
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        event = data.get("event")
                        if event == "start":
                            stream_sid = data.get("streamSid")
                            start_info = data.get("start", {})
                            current_call_sid = start_info.get("callSid")
                            logger.info(
                                "Twilio stream start streamSid=%s callSid=%s mediaFormat=%s",
                                stream_sid,
                                current_call_sid,
                                start_info.get("mediaFormat"),
                            )
                            if current_call_sid and current_call_sid not in CALL_TRANSCRIPTS:
                                CALL_TRANSCRIPTS[current_call_sid] = []
                            if current_call_sid and current_call_sid in CALL_CONTEXT:
                                context_text = CALL_CONTEXT.pop(current_call_sid)
                                logger.info(
                                    "Sending context to ElevenLabs for callSid=%s: %s",
                                    current_call_sid,
                                    context_text,
                                )
                                await eleven_ws.send(
                                    json.dumps(
                                        {
                                            "type": "contextual_update",
                                            "text": context_text,
                                        }
                                    )
                                )
                                logger.info(
                                    "Sent initial context to ElevenLabs for callSid=%s",
                                    current_call_sid,
                                )
                        elif event == "media":
                            payload = data.get("media", {}).get("payload")
                            if payload:
                                if input_format == "pcm_16000":
                                    payload = _twilio_ulaw_to_pcm16_16k(payload)
                                await eleven_ws.send(
                                    json.dumps({"user_audio_chunk": payload})
                                )
                        elif event == "stop":
                            logger.info("Twilio stream stop streamSid=%s", stream_sid)
                            break
                except WebSocketDisconnect:
                    logger.info("Twilio websocket disconnected")
                except Exception as exc:
                    logger.exception("Twilio->ElevenLabs stream error: %s", exc)
                finally:
                    try:
                        await eleven_ws.close()
                    except Exception:
                        pass

            async def forward_eleven_to_twilio() -> None:
                nonlocal input_format, output_format, current_call_sid
                try:
                    async for message in eleven_ws:
                        data = json.loads(message)
                        message_type = data.get("type")
                        if message_type == "ping":
                            event_id = data.get("ping_event", {}).get("event_id")
                            await eleven_ws.send(
                                json.dumps({"type": "pong", "event_id": event_id})
                            )
                            continue
                        if message_type == "conversation_initiation_metadata":
                            meta = data.get("conversation_initiation_metadata_event", {})
                            input_format = meta.get("user_input_audio_format") or input_format
                            output_format = (
                                meta.get("agent_output_audio_format") or output_format
                            )
                            logger.info(
                                "ElevenLabs metadata input=%s output=%s",
                                input_format,
                                output_format,
                            )
                            continue
                        if message_type == "user_transcript":
                            transcript = data.get("user_transcription_event", {}).get(
                                "user_transcript"
                            )
                            if transcript:
                                logger.info("User transcript: %s", transcript)
                                if current_call_sid and current_call_sid in CALL_TRANSCRIPTS:
                                    CALL_TRANSCRIPTS[current_call_sid].append(
                                        f"USER: {transcript}"
                                    )
                            continue
                        if message_type == "agent_response":
                            agent_text = data.get("agent_response_event", {}).get(
                                "agent_response"
                            )
                            if agent_text:
                                logger.info("Agent response: %s", agent_text)
                                if current_call_sid and current_call_sid in CALL_TRANSCRIPTS:
                                    CALL_TRANSCRIPTS[current_call_sid].append(
                                        f"AGENT: {agent_text}"
                                    )
                            continue
                        if message_type == "vad_score":
                            vad = data.get("vad_score_event", {}).get("vad_score")
                            logger.info("VAD score: %s", vad)
                            continue
                        if message_type == "audio":
                            audio_event: dict[str, Any] = data.get("audio_event", {})
                            audio_base_64 = audio_event.get("audio_base_64")
                            audio_format = audio_event.get("audio_format") or output_format
                            sample_rate = audio_event.get("sample_rate_hz")
                            if audio_base_64 and stream_sid:
                                twilio_payload = _to_twilio_ulaw_8000(
                                    audio_base_64,
                                    audio_format=audio_format,
                                    sample_rate=sample_rate,
                                )
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": twilio_payload},
                                        }
                                    )
                                )
                except Exception as exc:
                    logger.exception("ElevenLabs->Twilio stream error: %s", exc)
                finally:
                    try:
                        await websocket.close()
                    except Exception:
                        pass

            await asyncio.gather(forward_twilio_to_eleven(), forward_eleven_to_twilio())
    except Exception as exc:
        logger.exception("Twilio stream handler failed: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
