import json
from typing import Optional, List, Dict, Any, Tuple

from DBSVR import orchestrator

from tau_trait.agents.base import Agent
from tau_trait.envs.base import Env
from tau_trait.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


class ChatReActAgent(Agent):
    """
    tau-trait connector for the DBS-SVR orchestrator.

    The environment still sees exactly one Action per step. Internally, the
    orchestrator can run StateBuilder -> Budget -> Simulator -> Verifier -> Replan.
    """

    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        prompt_dir: str = "tau-trait/tau_trait/agents/DBSVR/Prompts",
        max_retries: int = 3,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.wiki = wiki
        self.prompt_dir = prompt_dir
        self.max_retries = max_retries

    def _parse_action(self, content: str) -> Dict[str, Any]:
        """Parse the final Action block returned by the DBS-SVR orchestrator."""
        action_str = content.split("Action:")[-1].strip() if "Action:" in content else content.strip()
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }

        if "name" not in action_parsed or "arguments" not in action_parsed:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: str(action_parsed)},
            }
        return action_parsed

    def generate_next_step(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Action, float]:
        message = orchestrator.orchestrate(
            observation=messages,
            model=self.model,
            provider=self.provider,
            temperature=self.temperature,
            tools_info=self.tools_info,
            wiki=self.wiki,
            prompt_dir=self.prompt_dir,
            max_retries=self.max_retries,
            respond_action_name=RESPOND_ACTION_NAME,
            respond_action_field_name=RESPOND_ACTION_FIELD_NAME,
        )

        action_parsed = self._parse_action(message.content)
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        cost = 0.0
        if hasattr(message, "metadata"):
            cost = float(message.metadata.get("cost", 0.0) or 0.0)
        return message.model_dump(), action, cost

    def solve(
        self,
        env: Env,
        task_index: Optional[int] = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info: Dict[str, Any] = {}

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

        info["total_internal_cost"] = total_cost
        return SolveResult(messages=messages, reward=reward, info=info)


REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that uses the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.
You should not use made-up or placeholder arguments.

If you need to respond to the user, use:
Action:
{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "your message"}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that uses the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.
You should not use made-up or placeholder arguments.

If you need to respond to the user, use:
Action:
{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "your message"}}}}

Try to be helpful and always follow the policy. Always generate valid JSON only.
"""
