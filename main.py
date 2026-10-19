import asyncio
from yapify import YapWrap
from yapify.agents import CallableAgent
from yapify.vad import EnergyVAD


async def demo_echo_agent(text: str):
    """Simple async generator agent."""
    yield f"You said: {text}. Yapify is working!"


def main():
    print("Yapify initialized successfully.")
    print("Example usage:")
    print("  from yapify import YapWrap")
    print("  yapper = YapWrap(agent=..., tts=..., stt=..., vad=...)")
    print("  await yapper.startlocal()")


if __name__ == "__main__":
    main()
