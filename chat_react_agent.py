import json
from litellm import completion

from tau_trait.agents.base import Agent
from tau_trait.envs.base import Env
from tau_trait.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)
from typing import Optional, List, Dict, Any, Tuple

from tau_trait.agents.DBSVR.orchestrator import DBSVRAgent


class ChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        prompt_dir: str = "tau_trait/agents/DBSVR/Prompts",
    ) -> None:
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info)
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info

        # Build the DBSVR orchestrator once. It receives tools_info and wiki
        # so its sub-agents (Simulator/Verifier/FinalOutput) can reason about
        # the available tools and the domain policy.
        self.dbsvr = DBSVRAgent(
            model=model,
            provider=provider,
            temperature=temperature,
            prompt_dir=prompt_dir,
            tools_info=tools_info,
            wiki=wiki,
        )

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float]:
        # ── Route through DBSVR instead of a single completion() call ──
        message = self.dbsvr.orchestrate(messages)
        action_str = message.content.split("Action:")[-1].strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            # this is a hack
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        # Cost is the sum of all sub-agent calls inside this orchestrate()
        cost = getattr(message, "response_cost", 0.0)
        return message.model_dump(), action, cost

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        for _ in range(max_num_steps):
            message, action, cost = self.generate_next_step(messages)
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            if response.done:
                break
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )
