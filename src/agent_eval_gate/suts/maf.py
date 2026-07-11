"""Microsoft Agent Framework (MAF) adapter."""

from __future__ import annotations

from typing import Optional

from agent_eval_gate.protocols import SUTOutput, Task
from agent_eval_gate.suts.base import make_sut_output


FRAMEWORK = "maf"


async def run(client, task: Task) -> SUTOutput:
    try:
        # MAF is the 2026 merger of AutoGen + Semantic Kernel.
        # We attempt the autogen-agentchat import first (predecessor merged into MAF).
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError:
            try:
                from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
                from semantic_kernel import Kernel
            except ImportError:
                raise ImportError("Neither autogen_agentchat nor semantic_kernel is installed.")

        import time
        start = time.time()

        # Use OpenAI-compatible path via autogen if available
        if "AssistantAgent" in dir():
            model_client = OpenAIChatCompletionClient(
                model=client.model,
                base_url=client.base_url,
                api_key=client.api_key,
            )
            agent = AssistantAgent(name="eval-agent", model_client=model_client, system_message="You are a helpful assistant.")
            result = await agent.run(task=task.prompt)
            raw_text = str(result)
        else:
            kernel = Kernel()
            service = OpenAIChatCompletion(
                ai_model_id=client.model,
                api_key=client.api_key,
                base_url=client.base_url,
            )
            kernel.add_service(service)
            # Simple chat via SK
            from semantic_kernel.functions.kernel_arguments import KernelArguments
            from semantic_kernel.kernel_pydantic import KernelBaseModel
            chat_function = kernel.add_function(
                plugin_name="eval",
                function_name="chat",
                prompt="{{$input}}",
            )
            result = await kernel.invoke(chat_function, KernelArguments(input=task.prompt))
            raw_text = str(result)

        latency_ms = (time.time() - start) * 1000.0
        return make_sut_output(task, FRAMEWORK, client.model, raw_text, latency_ms=latency_ms)
    except Exception as exc:
        raise RuntimeError(f"Microsoft Agent Framework adapter failed: {exc}") from exc
