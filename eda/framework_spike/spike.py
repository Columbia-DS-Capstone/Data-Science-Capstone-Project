"""Two-stage orchestrator spike: proposal -> quality gate -> capped retry."""

import argparse
import json
import os
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

TARGET = 0.6


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt: int = Field(ge=1)
    quality: float = Field(ge=0, le=1)


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str


class Decision(BaseModel):
    stage: str
    attempt: int
    outcome: str
    reason: str


class RunState(TypedDict):
    max_attempts: int
    attempts: int
    proposal: Proposal | None
    gate: Gate | None
    decisions: list[Decision]


def proposal_for(attempt: int) -> Proposal:
    return Proposal(attempt=attempt, quality=0.4 if attempt == 1 else 0.7)


def gate_for(proposal: Proposal) -> Gate:
    passed = proposal.quality >= TARGET
    return Gate(passed=passed, reason="quality meets target" if passed else "quality below target")


def checked_gate(proposal: Proposal, gate: Gate) -> Gate:
    if gate.passed != (proposal.quality >= TARGET):
        raise ValueError("Gate decision disagrees with the deterministic quality rule")
    return gate


def initial_state(max_attempts: int) -> RunState:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    return {"max_attempts": max_attempts, "attempts": 0, "proposal": None,
            "gate": None, "decisions": []}


def run_langgraph(
    propose: Callable[[int], Proposal] = proposal_for,
    judge: Callable[[Proposal], Gate] = gate_for,
    max_attempts: int = 2,
) -> RunState:
    from langgraph.graph import END, START, StateGraph

    def proposal_node(state: RunState) -> dict:
        attempt = state["attempts"] + 1
        proposal = Proposal.model_validate(propose(attempt))
        if proposal.attempt != attempt:
            raise ValueError("Proposal attempt does not match run state")
        return {"attempts": attempt, "proposal": proposal,
                "decisions": [*state["decisions"], Decision(
                    stage="proposal", attempt=attempt, outcome="created",
                    reason=f"quality={proposal.quality:.2f}")]}

    def gate_node(state: RunState) -> dict:
        proposal = state["proposal"]
        assert proposal is not None
        gate = checked_gate(proposal, Gate.model_validate(judge(proposal)))
        return {"gate": gate, "decisions": [*state["decisions"], Decision(
            stage="gate", attempt=state["attempts"],
            outcome="pass" if gate.passed else "fail", reason=gate.reason)]}

    def next_step(state: RunState) -> str:
        gate = state["gate"]
        assert gate is not None
        if not gate.passed and state["attempts"] < state["max_attempts"]:
            return "proposal"
        return END

    graph = StateGraph(RunState)
    graph.add_node("proposal", proposal_node)
    graph.add_node("gate", gate_node)
    graph.add_edge(START, "proposal")
    graph.add_edge("proposal", "gate")
    graph.add_conditional_edges("gate", next_step)
    return graph.compile().invoke(initial_state(max_attempts),
                                  config={"recursion_limit": max(25, 2 * max_attempts + 2)})


def run_agents_sdk(
    model, max_attempts: int = 2,
    quality_for_attempt: Callable[[int], float] | None = None,
    context: str = "",
    on_usage: Callable[[object], None] | None = None,
) -> RunState:
    from agents import Agent, ModelSettings, Runner

    proposal_agent = Agent(
        name="Proposal", model=model, output_type=Proposal,
        model_settings=ModelSettings(temperature=0),
        instructions="Copy the requested attempt and quality exactly into the structured output.",
    )
    gate_agent = Agent(
        name="Quality gate", model=model, output_type=Gate,
        model_settings=ModelSettings(temperature=0),
        instructions="Return passed=true exactly when quality >= 0.6. Give a short reason.",
    )
    state = initial_state(max_attempts)
    for attempt in range(1, max_attempts + 1):
        quality = (quality_for_attempt(attempt) if quality_for_attempt
                   else proposal_for(attempt).quality)
        result = Runner.run_sync(
            proposal_agent, f"{context} Return attempt={attempt}; quality={quality}.")
        if on_usage:
            on_usage(result.context_wrapper.usage)
        proposal = Proposal.model_validate(result.final_output)
        if proposal.attempt != attempt:
            raise ValueError("Proposal attempt does not match run state")
        if proposal.quality != quality:
            raise ValueError("Proposal quality does not match the requested value")
        state["attempts"] = attempt
        state["proposal"] = proposal
        state["decisions"].append(Decision(
            stage="proposal", attempt=attempt, outcome="created",
            reason=f"quality={proposal.quality:.2f}"))
        result = Runner.run_sync(
            gate_agent, f"{context} Quality={proposal.quality}; target={TARGET}.")
        if on_usage:
            on_usage(result.context_wrapper.usage)
        gate = Gate.model_validate(result.final_output)
        state["gate"] = checked_gate(proposal, gate)
        state["decisions"].append(Decision(
            stage="gate", attempt=attempt, outcome="pass" if gate.passed else "fail",
            reason=gate.reason))
        if gate.passed:
            break
    return state


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        parts = shlex.split(line, comments=True)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) == 1 and "=" in parts[0]:
            key, value = parts[0].split("=", 1)
            if key in {"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "VERTEX_MODEL",
                       "GOOGLE_APPLICATION_CREDENTIALS"}:
                os.environ.setdefault(key, value)


def vertex_clients():
    """ADC supplies the short-lived token; project and location must be explicit."""
    from google.auth import default
    from google.auth.transport.requests import Request
    from openai import AsyncOpenAI, OpenAI

    load_local_env(Path(__file__).with_name(".env"))
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION for the team project")
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    host = ("aiplatform.googleapis.com" if location == "global"
            else f"{location}-aiplatform.googleapis.com")
    base_url = f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
    model = os.environ.get("VERTEX_MODEL", "google/gemini-3.5-flash")
    return OpenAI(base_url=base_url, api_key=credentials.token), AsyncOpenAI(
        base_url=base_url, api_key=credentials.token), model


def vertex_stages(client, model: str, quality_for_attempt=None, context="", on_usage=None):
    def structured(schema: type[BaseModel], prompt: str):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
            temperature=0,
        )
        if on_usage:
            on_usage(response.usage)
        return schema.model_validate_json(response.choices[0].message.content)

    def propose(attempt: int) -> Proposal:
        quality = (quality_for_attempt(attempt) if quality_for_attempt
                   else proposal_for(attempt).quality)
        proposal = structured(
            Proposal, f"{context} Return attempt={attempt}; quality={quality}.")
        if proposal.quality != quality:
            raise ValueError("Proposal quality does not match the requested value")
        return proposal

    def judge(proposal: Proposal) -> Gate:
        return structured(
            Gate, f"{context} Pass exactly if quality {proposal.quality} >= {TARGET}; "
                  "give a short reason.")

    return propose, judge


def fake_sdk_model(quality_for_attempt: Callable[[int], float] | None = None):
    """Local HTTP fixture exercises the real Agents SDK without a model bill."""
    import httpx
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        attempt = (calls + 1) // 2
        proposal = (Proposal(attempt=attempt, quality=quality_for_attempt(attempt))
                    if quality_for_attempt else proposal_for(attempt))
        output = proposal if calls % 2 else gate_for(proposal)
        return httpx.Response(200, json={
            "id": f"chatcmpl-{calls}", "object": "chat.completion", "created": 0,
            "model": "offline", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": output.model_dump_json()}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = AsyncOpenAI(api_key="offline", base_url="https://offline.test/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    return OpenAIChatCompletionsModel(model="offline", openai_client=client)


def summary(state: RunState) -> dict:
    return {"attempts": state["attempts"], "passed": state["gate"].passed,
            "decisions": [decision.model_dump() for decision in state["decisions"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertex", action="store_true", help="Use Gemini on Vertex AI via ADC")
    parser.add_argument("--check-vertex", action="store_true",
                        help="Make one structured Vertex call")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--log-dir", type=Path, help="Write one decisions.jsonl per framework")
    args = parser.parse_args()
    if args.vertex or args.check_vertex:
        from agents import OpenAIChatCompletionsModel, set_tracing_disabled
        try:
            client, async_client, model_name = vertex_clients()
        except RuntimeError as exc:
            parser.error(str(exc))
        propose, judge = vertex_stages(client, model_name)
        if args.check_vertex:
            print(json.dumps({"vertex_structured_call": "ok", "model": model_name,
                              "proposal": propose(1).model_dump()}))
            return
        sdk_model = OpenAIChatCompletionsModel(model=model_name, openai_client=async_client)
        set_tracing_disabled(True)
    else:
        propose, judge, sdk_model = proposal_for, gate_for, fake_sdk_model()
        from agents import set_tracing_disabled
        set_tracing_disabled(True)
    runs = {"langgraph": run_langgraph(propose, judge, args.max_attempts),
            "agents_sdk": run_agents_sdk(sdk_model, args.max_attempts)}
    for name, state in runs.items():
        print(json.dumps({"framework": name, **summary(state)}))
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            with (args.log_dir / f"{name}_decisions.jsonl").open("w") as log:
                for decision in state["decisions"]:
                    log.write(decision.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
