"""SUT registry — maps framework name to adapter."""

from __future__ import annotations

from typing import Callable

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.llm_client import LLMClient
from agent_eval_gate.suts.openai_sdk import run as openai_sdk_run
from agent_eval_gate.suts.pydantic_ai import run as pydantic_ai_run
from agent_eval_gate.suts.adk import run as adk_run
from agent_eval_gate.suts.strands import run as strands_run
from agent_eval_gate.suts.langchain import run as langchain_run
from agent_eval_gate.suts.claude_agent_sdk import run as claude_agent_sdk_run
from agent_eval_gate.suts.maf import run as maf_run
from agent_eval_gate.suts.crewai import run as crewai_run
from agent_eval_gate.suts.smolagents import run as smolagents_run
from agent_eval_gate.suts.react_loop import run as react_loop_run


SUT_REGISTRY: dict[str, Callable[[LLMClient, Task], "SUTOutput"]] = {
    "openai_sdk": openai_sdk_run,
    "pydantic_ai": pydantic_ai_run,
    "adk": adk_run,
    "strands": strands_run,
    "langchain": langchain_run,
    "claude_agent_sdk": claude_agent_sdk_run,
    "maf": maf_run,
    "crewai": crewai_run,
    "smolagents": smolagents_run,
    "react_loop": react_loop_run,
}

BASELINE_SUTS = {"react_loop"}
