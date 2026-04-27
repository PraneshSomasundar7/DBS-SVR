from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm import completion


@dataclass
class DBSVRMessage:
    """Small message object compatible with LiteLLM-style usage in the connector."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> Dict[str, Any]:
        data = {"role": self.role, "content": self.content}
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class DBSVRAgent:
    """
    DBS-SVR orchestrator for tau-trait/tau-bench-style environments.

    The benchmark environment must execute tools. This orchestrator only returns a final
    Action block:

    Action:
    {"name": "<tool_or_respond>", "arguments": {...}}
    """

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
        prompt_dir: str = "tau_trait/agents/DBSSVR/Prompts",
        max_retries: int = 3,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        wiki: str = "",
        respond_action_name: str = "respond",
        respond_action_field_name: str = "content",
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.prompt_dir = Path(prompt_dir)
        self.max_retries = max_retries
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.respond_action_name = respond_action_name
        self.respond_action_field_name = respond_action_field_name
        self.total_cost = 0.0

    # ------------------------------------------------------------------
    # Prompt and LLM helpers
    # ------------------------------------------------------------------
    def _prompt_file(self, *candidates: str) -> str:
        """Return the first prompt filename that exists."""
        for filename in candidates:
            if (self.prompt_dir / filename).exists():
                return filename
        return candidates[0]

    def load_prompt(self, filename: str) -> Dict[str, Any]:
        prompt_path = self.prompt_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = yaml.safe_load(file)

        if not isinstance(prompt, dict):
            raise ValueError(f"Prompt file must contain a YAML object: {prompt_path}")
        if "system" not in prompt:
            raise ValueError(f"Prompt file must contain a 'system' key: {prompt_path}")
        return prompt

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        rendered = template
        for key, value in variables.items():
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            rendered = rendered.replace("{{ " + key + " }}", value)
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered

    def build_messages(self, observation: str, prompt: Dict[str, Any]) -> List[Dict[str, str]]:
        variables = {
            "observation": observation,
            "tools_info": self.tools_info,
            "wiki": self.wiki,
            "respond_action_name": self.respond_action_name,
            "respond_action_field_name": self.respond_action_field_name,
        }
        system_prompt = self._render_template(prompt["system"], variables)
        user_template = prompt.get("user", "{{ observation }}")
        user_content = self._render_template(user_template, variables)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _response_cost(self, response: Any) -> float:
        hidden = getattr(response, "_hidden_params", {}) or {}
        try:
            return float(hidden.get("response_cost", 0.0) or 0.0)
        except Exception:
            return 0.0

    def chat(self, messages: List[Dict[str, str]]) -> DBSVRMessage:
        response = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        self.total_cost += self._response_cost(response)
        msg = response.choices[0].message
        return DBSVRMessage(role="assistant", content=msg.content or "")

    def run_agent(self, observation: Union[str, Dict[str, Any]], prompt_file: str) -> DBSVRMessage:
        if not isinstance(observation, str):
            observation = json.dumps(observation, ensure_ascii=False, indent=2)
        prompt = self.load_prompt(prompt_file)
        messages = self.build_messages(observation, prompt)
        return self.chat(messages)

    # ------------------------------------------------------------------
    # JSON/action helpers
    # ------------------------------------------------------------------
    def safe_json_loads(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"parse_error": True, "raw_output": text}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {"parse_error": True, "raw_output": text}

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "verified", "accepted", "execute"}
        return False

    def normalize_budget(self, budget_output: Union[str, Dict[str, Any]]) -> str:
        if isinstance(budget_output, dict):
            raw = str(budget_output.get("budget", ""))
        else:
            parsed = self.safe_json_loads(str(budget_output))
            if not parsed.get("parse_error") and "budget" in parsed:
                raw = str(parsed.get("budget", ""))
            else:
                raw = str(budget_output)

        raw_lower = raw.strip().lower()
        if "low" in raw_lower:
            return "LOW"
        if "high" in raw_lower:
            return "HIGH"
        return "HIGH"

    def normalize_candidate_action(self, simulation_or_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts either:
        {"candidate_action": {"tool_name": "x", "arguments": {}, "draft_message": null}}
        or
        {"candidate_action": {"name": "x", "arguments": {}}}
        or
        {"name": "x", "arguments": {}}
        and returns {"name": ..., "arguments": {...}}.
        """
        candidate = simulation_or_action.get("candidate_action", simulation_or_action)
        if not isinstance(candidate, dict):
            return self.respond_action("I need more information before I can continue safely.")

        if "name" in candidate:
            return {
                "name": candidate.get("name") or self.respond_action_name,
                "arguments": candidate.get("arguments") or {},
            }

        tool_name = candidate.get("tool_name")
        draft_message = candidate.get("draft_message")
        arguments = candidate.get("arguments") or {}

        if tool_name:
            return {"name": tool_name, "arguments": arguments}

        if draft_message:
            return self.respond_action(str(draft_message))

        return self.respond_action("I need more information before I can continue safely.")

    def respond_action(self, message: str) -> Dict[str, Any]:
        return {
            "name": self.respond_action_name,
            "arguments": {self.respond_action_field_name: message},
        }

    def action_block(self, action: Dict[str, Any]) -> str:
        return "Action:\n" + json.dumps(action, ensure_ascii=False)

    def ensure_action_message(self, text: str) -> DBSVRMessage:
        """Make sure final model output can be parsed by the tau connector."""
        text = (text or "").strip()
        if "Action:" in text:
            action_part = text.split("Action:")[-1].strip()
            parsed = self.safe_json_loads(action_part)
            if not parsed.get("parse_error") and "name" in parsed and "arguments" in parsed:
                return DBSVRMessage(role="assistant", content="Action:\n" + json.dumps(parsed, ensure_ascii=False))

        parsed = self.safe_json_loads(text)
        if not parsed.get("parse_error") and "name" in parsed and "arguments" in parsed:
            return DBSVRMessage(role="assistant", content=self.action_block(parsed))

        return DBSVRMessage(role="assistant", content=self.action_block(self.respond_action(text)))

    def build_initial_payload(self, observation: Union[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(observation, list):
            messages = observation
            latest_user_observation = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    latest_user_observation = str(msg.get("content", ""))
                    break
            return {
                "latest_observation": latest_user_observation,
                "conversation_messages": messages,
                "wiki": self.wiki,
                "available_tools": self.tools_info,
            }

        return {
            "latest_observation": str(observation),
            "conversation_messages": [{"role": "user", "content": str(observation)}],
            "wiki": self.wiki,
            "available_tools": self.tools_info,
        }

    def apply_replan_update(self, state: Dict[str, Any], replan_update: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(state)

        if not isinstance(replan_update, dict) or replan_update.get("parse_error"):
            updated["last_replan_intent"] = "ASK_USER"
            updated["last_replan_reason"] = "Replanner returned invalid JSON."
            updated.setdefault("missing_facts", [])
            return updated

        if "revised_subgoal" in replan_update:
            updated["current_subgoal"] = replan_update["revised_subgoal"]
        if "current_subgoal" in replan_update:
            updated["current_subgoal"] = replan_update["current_subgoal"]

        if "missing_facts" in replan_update:
            updated["missing_facts"] = replan_update.get("missing_facts") or []

        if "known_facts" in replan_update:
            known = updated.get("known_facts", {})
            if isinstance(known, dict) and isinstance(replan_update["known_facts"], dict):
                known.update(replan_update["known_facts"])
                updated["known_facts"] = known

        if "user_constraints" in replan_update:
            updated["user_constraints"] = replan_update.get("user_constraints") or updated.get("user_constraints", [])

        updated["last_replan_intent"] = replan_update.get("replan_intent")
        updated["last_replan_reason"] = replan_update.get("replan_reason")
        return updated

    def missing_info_message(self, state: Dict[str, Any]) -> str:
        missing = state.get("missing_facts") or []
        if missing:
            return "I need the following information before I can continue safely: " + ", ".join(map(str, missing))
        return "I need more information before I can continue safely."

    # ------------------------------------------------------------------
    # Main DBS-SVR flow
    # ------------------------------------------------------------------
    def orchestrate(self, observation: Union[str, List[Dict[str, Any]]]) -> DBSVRMessage:
        self.total_cost = 0.0
        trace: Dict[str, Any] = {"steps": []}
        initial_payload = self.build_initial_payload(observation)

        # 1. State Builder
        state_msg = self.run_agent(initial_payload, self._prompt_file("StateBuilder.yaml"))
        state = self.safe_json_loads(state_msg.content)
        if state.get("parse_error"):
            state = {
                "user_goal": "Handle the current user request.",
                "current_subgoal": "Recover from invalid state builder output.",
                "known_facts": {},
                "missing_facts": [],
                "user_constraints": [],
                "risk_level": "high",
                "intent_type": "unknown",
                "raw_state_builder_output": state_msg.content,
            }
        state["conversation_messages"] = initial_payload["conversation_messages"]
        state["latest_observation"] = initial_payload["latest_observation"]
        state["available_tools"] = self.tools_info
        state["wiki"] = self.wiki
        trace["state_builder"] = state

        # 2. Budget Router
        budget_file = self._prompt_file("BudgetScope.yaml", "BudgetAllocator.yaml")
        budget_msg = self.run_agent({"state": state}, budget_file)
        budget_json = self.safe_json_loads(budget_msg.content)
        budget = self.normalize_budget(budget_json if not budget_json.get("parse_error") else budget_msg.content)
        state["budget"] = budget
        trace["budget"] = {"raw": budget_msg.content, "normalized": budget}

        # 3. LOW budget path: directly produce final Action block.
        if budget == "LOW":
            final_payload = {
                "budget": "LOW",
                "state_builder_output": state,
                "simulator_output": None,
                "verifier_output": None,
                "final_candidate_action": None,
                "available_tools": self.tools_info,
                "respond_action_name": self.respond_action_name,
                "respond_action_field_name": self.respond_action_field_name,
                "instruction": "LOW budget selected. Produce the immediate final Action block.",
            }
            final_msg = self.run_agent(final_payload, self._prompt_file("FinalOutput.yaml"))
            output = self.ensure_action_message(final_msg.content)
            output.metadata = {"trace": trace, "cost": self.total_cost}
            return output

        # 4. HIGH budget path: simulate -> verify -> replan loop.
        last_simulation: Optional[Dict[str, Any]] = None
        last_verification: Optional[Dict[str, Any]] = None
        replan_history: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 1):
            simulator_payload = {
                "state": state,
                "attempt_number": attempt,
                "max_retries": self.max_retries,
                "available_tools": self.tools_info,
                "instruction": "Simulate exactly one candidate action. Do not execute it.",
            }
            sim_msg = self.run_agent(simulator_payload, self._prompt_file("Simulator.yaml"))
            simulation = self.safe_json_loads(sim_msg.content)
            last_simulation = simulation

            if simulation.get("parse_error"):
                replan_history.append({"attempt": attempt, "failure": "simulator_parse_error", "raw": sim_msg.content})
                state["current_subgoal"] = "Produce a valid JSON candidate action."
                continue

            candidate_action = self.normalize_candidate_action(simulation)

            verifier_payload = {
                "state": state,
                "candidate_action": candidate_action,
                "simulator_output": simulation,
                "attempt_number": attempt,
                "available_tools": self.tools_info,
                "instruction": "Verify whether the candidate action can safely be returned to the tau environment.",
            }
            verify_msg = self.run_agent(verifier_payload, self._prompt_file("Verifier.yaml"))
            verification = self.safe_json_loads(verify_msg.content)
            last_verification = verification

            accepted = self._as_bool(verification.get("accepted")) or verify_msg.content.strip().lower() == "verified"
            if accepted:
                verified_action = verification.get("final_candidate_action") or candidate_action
                verified_action = self.normalize_candidate_action(verified_action)
                final_payload = {
                    "budget": "HIGH",
                    "state_builder_output": state,
                    "simulator_output": simulation,
                    "verifier_output": verification,
                    "final_candidate_action": verified_action,
                    "available_tools": self.tools_info,
                    "respond_action_name": self.respond_action_name,
                    "respond_action_field_name": self.respond_action_field_name,
                    "instruction": "Verifier accepted the candidate. Produce the final Action block.",
                }
                final_msg = self.run_agent(final_payload, self._prompt_file("FinalOutput.yaml"))
                output = self.ensure_action_message(final_msg.content)
                trace["svr"] = {"accepted_attempt": attempt, "replans": replan_history}
                output.metadata = {"trace": trace, "cost": self.total_cost}
                return output

            # Replan on failed verification.
            replan_file = self._prompt_file("Replan.yaml", "Replanner.yaml")
            replan_payload = {
                "state": state,
                "failed_candidate_action": candidate_action,
                "simulator_output": simulation,
                "verification": verification,
                "attempt_number": attempt,
                "max_retries": self.max_retries,
                "available_tools": self.tools_info,
                "instruction": "Revise the state so the next simulation attempt changes something.",
            }
            replan_msg = self.run_agent(replan_payload, replan_file)
            replan_update = self.safe_json_loads(replan_msg.content)
            state = self.apply_replan_update(state, replan_update)
            replan_history.append(
                {
                    "attempt": attempt,
                    "candidate_action": candidate_action,
                    "verification": verification,
                    "replan_update": replan_update,
                }
            )

            if state.get("last_replan_intent") == "ASK_USER":
                return self._final_response_from_action(
                    state=state,
                    trace=trace,
                    simulation=last_simulation,
                    verification=last_verification,
                    action=self.respond_action(self.missing_info_message(state)),
                    reason="Replanner requested ASK_USER.",
                )

            if state.get("last_replan_intent") == "ESCALATE":
                return self._final_response_from_action(
                    state=state,
                    trace=trace,
                    simulation=last_simulation,
                    verification=last_verification,
                    action=self.respond_action("I need human assistance before I can safely continue."),
                    reason="Replanner requested ESCALATE.",
                )

        # 5. Fallback after max retries.
        return self._final_response_from_action(
            state=state,
            trace=trace,
            simulation=last_simulation,
            verification=last_verification,
            action=self.respond_action(
                "I could not find a verified safe action after multiple attempts. I need more information or human assistance."
            ),
            reason="Maximum DBS-SVR replans exceeded.",
        )

    def _final_response_from_action(
        self,
        state: Dict[str, Any],
        trace: Dict[str, Any],
        simulation: Optional[Dict[str, Any]],
        verification: Optional[Dict[str, Any]],
        action: Dict[str, Any],
        reason: str,
    ) -> DBSVRMessage:
        final_payload = {
            "budget": state.get("budget", "HIGH"),
            "state_builder_output": state,
            "simulator_output": simulation,
            "verifier_output": verification,
            "final_candidate_action": action,
            "available_tools": self.tools_info,
            "respond_action_name": self.respond_action_name,
            "respond_action_field_name": self.respond_action_field_name,
            "instruction": reason,
        }
        final_msg = self.run_agent(final_payload, self._prompt_file("FinalOutput.yaml"))
        output = self.ensure_action_message(final_msg.content)
        trace["fallback_reason"] = reason
        output.metadata = {"trace": trace, "cost": self.total_cost}
        return output


def orchestrate(
    observation: Union[str, List[Dict[str, Any]]],
    model: str,
    provider: str,
    temperature: float = 0.0,
    tools_info: Optional[List[Dict[str, Any]]] = None,
    wiki: str = "",
    prompt_dir: str = "tau-trait/tau_trait/agents/DBSSVR/Prompts",
    max_retries: int = 3,
    respond_action_name: str = "respond",
    respond_action_field_name: str = "content",
) -> DBSVRMessage:
    agent = DBSVRAgent(
        model=model,
        provider=provider,
        temperature=temperature,
        prompt_dir=prompt_dir,
        max_retries=max_retries,
        tools_info=tools_info,
        wiki=wiki,
        respond_action_name=respond_action_name,
        respond_action_field_name=respond_action_field_name,
    )
    return agent.orchestrate(observation)
