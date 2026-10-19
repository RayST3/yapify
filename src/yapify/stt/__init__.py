"""Speech-to-Text (STT) streaming transcription providers for yapify."""

from yapify.core.base import BaseSTT
from yapify.stt.deepgram import DeepgramSTT
from yapify.stt.assemblyai import AssemblyAISTT
from yapify.stt.openai_realtime import OpenAIRealtimeSTT
from yapify.stt.elevenlabs import ElevenLabsSTT
from yapify.stt.speechmatics import SpeechmaticsSTT
from yapify.stt.soniox import SonioxSTT
from yapify.stt.gladia import GladiaSTT
from yapify.stt.google_chirp import GoogleChirpSTT

__all__ = [
    "BaseSTT",
    "DeepgramSTT",
    "AssemblyAISTT",
    "OpenAIRealtimeSTT",
    "ElevenLabsSTT",
    "SpeechmaticsSTT",
    "SonioxSTT",
    "GladiaSTT",
    "GoogleChirpSTT",
]

