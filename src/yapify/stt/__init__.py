"""Speech-to-Text (STT) streaming transcription providers for yapify."""

from yapify.stt.deepgram import DeepgramSTT
from yapify.stt.assemblyai import AssemblyAISTT
from yapify.stt.openai_realtime import OpenAIRealtimeSTT
from yapify.stt.elevenlabs import ElevenLabsSTT
from yapify.stt.speechmatics import SpeechmaticsSTT
from yapify.stt.soniox import SonioxSTT
from yapify.stt.gladia import GladiaSTT
from yapify.stt.google_chirp import GoogleChirpSTT
from yapify.stt.azure import AzureSpeechSTT
from yapify.stt.aws_transcribe import AWSTranscribeSTT

__all__ = [
    "DeepgramSTT",
    "AssemblyAISTT",
    "OpenAIRealtimeSTT",
    "ElevenLabsSTT",
    "SpeechmaticsSTT",
    "SonioxSTT",
    "GladiaSTT",
    "GoogleChirpSTT",
    "AzureSpeechSTT",
    "AWSTranscribeSTT",
]

