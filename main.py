import asyncio
from yapwrap import YapWrap
from yapwrap.agents import CallableAgent
from yapwrap.vad import EnergyVAD


async def demo_echo_agent(text: str):
    """Simple async generator agent."""
    yield f"You said: {text}. YapWrap is working!"


def main():
    print("YapWrap initialized successfully.")
    print("Example usage:")
    print("  from yapwrap import YapWrap")
    print("  yapper = YapWrap(agent=..., tts=..., stt=..., vad=...)")
    print("  await yapper.startlocal()")


if __name__ == "__main__":
    main()
