"""Fresh Agno/Gemini voice components for each transport connection."""

from examples._shared import make_yapper
from examples.frameworks.agno_agent import create_agent


def create_yapper():
    return make_yapper(create_agent())
