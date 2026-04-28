from litellm import completion
import yaml
from pathlib import Path
import json
import time
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

# ── ANSI colours (gracefully disabled if the terminal doesn't support them) ──
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "grey":   "\033[90m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "magenta":"\033[95m",
}

def _c(colour: str, text: str) -> str:
    return f"{_C.get(colour, '')}{text}{_C['reset']}"


class DBSVRAgent:
    """
    DBS-SVR test-time scaling orchestrator.

    Pipeline per call to orchestrate():
        StateBuilder -> BudgetScope
            -> (LOW)  FinalOutput
            -> (HIGH) [Simulator -> Verifier -> (Replan)]* -> FinalOutput

    Each sub-agent is a *stateless* [system, user] LLM call.
    Inputs flow forward as plain strings; nothing is mutated in place.

    Returns a litellm-compatible message object whose `.content`
    contains a string of the form:

        Action:
        {"name": "...", "arguments": {...}}

    so that ChatReActAgent.generate_next_step can parse it directly.
    """

    MAX_REPLAN_RETRIES = 3

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.0,
        prompt_dir: str = "tau_trait/agents/DBSVR/Prompts",
        tools_info: Optional[List[Dict[str, Any]]] = None,
        wiki: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.prompt_dir = Path(prompt_dir)
        self.tools_info = tools_info or []
        self.wiki = wiki or ""
        self.verbose = verbose
        # Accumulated cost for the current orchestrate() call.
        self._call_cost: float = 0.0
        # Stuck-loop throttle (persists across orchestrate() calls).
        self._last_stuck_turn: int = -10  # turn index when detector last fired
        self._stuck_fire_count: int = 0  # how many times it has fired this episode

    # --------------------------------------------------------------------- #
    # Logging helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _divider(char: str = "─", width: int = 70) -> str:
        return char * width

    def _log(self, label: str, colour: str, body: str = "", indent: int = 0) -> None:
        """Print a labelled progress line. body is printed indented below."""
        pad = "  " * indent
        print(f"{pad}{_c(colour, label)}")
        if body:
            for line in body.splitlines():
                print(f"{pad}  {_c('grey', line)}")

    def _log_json(self, label: str, colour: str, raw: str, indent: int = 0) -> None:
        """Pretty-print a JSON blob under a label, truncating if very long."""
        try:
            obj = json.loads(raw.strip())
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pretty = raw
        # Truncate large blobs so the terminal stays readable
        lines = pretty.splitlines()
        if len(lines) > 30:
            pretty = "\n".join(lines[:30]) + f"\n  ... ({len(lines)-30} more lines)"
        self._log(label, colour, pretty, indent=indent)

    # --------------------------------------------------------------------- #
    # Low-level helpers
    # --------------------------------------------------------------------- #
    def load_prompt(self, filename: str) -> str:
        """Return the `system:` value of a YAML prompt file."""
        path = self.prompt_dir / filename
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "system" not in data:
            raise ValueError(f"{filename} must contain a top-level 'system:' key")
        return data["system"]

    def chat(self, system_prompt: str, user_input: str) -> str:
        """Single isolated [system, user] LLM call. Returns plain string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        response = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        # Accumulate cost across all sub-agent calls within one orchestrate()
        try:
            self._call_cost += float(
                response._hidden_params.get("response_cost", 0.0) or 0.0
            )
        except Exception:
            pass
        return response.choices[0].message.content

    def run_agent(
        self,
        user_input: str,
        prompt_file: str,
        extra_system: Optional[str] = None,
    ) -> str:
        """
        Load a prompt YAML, optionally append runtime context to the system
        prompt, send a single [system, user] pair, return the assistant
        string.
        """
        system_prompt = self.load_prompt(prompt_file)
        if extra_system:
            system_prompt = system_prompt + "\n\n" + extra_system

        t0 = time.perf_counter()
        out = self.chat(system_prompt, user_input)
        elapsed = time.perf_counter() - t0

        if self.verbose:
            label = f"  ↳ {prompt_file}  ({elapsed:.2f}s)"
            self._log_json(label, "grey", out, indent=1)
        return out

    # --------------------------------------------------------------------- #
    # Format / parse helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _humanize_assistant_content(content: str) -> str:
        """
        An assistant message in the main trajectory looks like:
            "Action:\n{\"name\": \"...\", \"arguments\": {...}}"
        Convert that into plain text the StateBuilder can read:
          - respond actions       -> the user-facing message
          - tool calls            -> "[called <tool>(<args>)]"
          - anything unparseable  -> returned as-is
        """
        if not isinstance(content, str) or "Action:" not in content:
            return content if isinstance(content, str) else json.dumps(content)
        action_str = content.split("Action:")[-1].strip()
        try:
            action = json.loads(action_str)
        except (json.JSONDecodeError, TypeError):
            return content
        if not isinstance(action, dict):
            return content
        name = action.get("name", "")
        args = action.get("arguments", {}) or {}
        # tau_bench/tau_trait respond action: name == "respond" and the
        # message lives under arguments.content (RESPOND_ACTION_FIELD_NAME).
        if name == "respond" and isinstance(args, dict):
            for key in ("content", "message", "response"):
                if key in args:
                    return str(args[key])
            return json.dumps(args, ensure_ascii=False)
        # Tool call
        return f"[called tool `{name}` with arguments {json.dumps(args, ensure_ascii=False)}]"

    @classmethod
    def _format_main_traj(cls, messages: List[Dict[str, Any]]) -> str:
        """
        Flatten the main trajectory into a clean conversation that
        StateBuilder can reliably parse:
          - the giant system prompt (wiki + tools + ReAct instruction) is
            dropped here because the wiki is injected via extra_system and
            the tool list is already in the Simulator/Verifier system
            prompt. Including it here just buries the actual conversation.
          - assistant turns are humanized (see _humanize_assistant_content)
          - user turns prefixed with "API output:" are relabelled as
            TOOL RESULT so the StateBuilder doesn't think the user said
            "API output: ...".
        """
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            if not isinstance(content, str):
                content = str(content)

            if role == "system":
                # Skip — too noisy for the StateBuilder. Wiki is passed
                # separately via extra_system.
                continue

            if role == "assistant":
                lines.append(f"AGENT: {cls._humanize_assistant_content(content)}")
            elif role == "user":
                if content.startswith("API output:"):
                    lines.append(
                        f"TOOL RESULT: {content[len('API output:'):].strip()}"
                    )
                else:
                    lines.append(f"USER: {content}")
            else:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    @classmethod
    def _recent_traj(cls, messages: List[Dict[str, Any]], max_lines: int = 12) -> str:
        """
        Last few humanized lines of the trajectory — used to give the
        Simulator / Verifier / Replanner immediate context about what just
        happened, so they don't re-propose the same action.
        """
        full = cls._format_main_traj(messages)
        if not full:
            return "(empty)"
        lines = full.splitlines()
        return "\n".join(lines[-max_lines:])

    @staticmethod
    def _safe_json_loads(s: str, default: Any) -> Any:
        try:
            return json.loads(s.strip())
        except (json.JSONDecodeError, AttributeError, TypeError):
            return default

    def _tools_block(self) -> str:
        """Runtime context block injected into Simulator/Verifier."""
        return (
            "## Domain policy (wiki)\n"
            f"{self.wiki}\n\n"
            "## Available tools\n"
            f"{json.dumps(self.tools_info, indent=2)}"
        )

    @staticmethod
    def _merge_state(state_json: str, replan_json: str) -> str:
        """
        Apply replanner updates on top of the prior state.
        Returns a new state JSON string.
        """
        state = DBSVRAgent._safe_json_loads(state_json, {})
        replan = DBSVRAgent._safe_json_loads(replan_json, {})

        if not isinstance(state, dict):
            state = {}

        # current_subgoal <- revised_subgoal (if changed)
        if replan.get("revised_subgoal"):
            state["current_subgoal"] = replan["revised_subgoal"]

        # union missing_facts
        if isinstance(replan.get("missing_facts"), list):
            existing = state.get("missing_facts") or []
            state["missing_facts"] = list({*existing, *replan["missing_facts"]})

        # merge known_facts dict
        if isinstance(replan.get("known_facts"), dict):
            kf = state.get("known_facts") or {}
            kf.update(replan["known_facts"])
            state["known_facts"] = kf

        # extend user_constraints
        if isinstance(replan.get("user_constraints"), list):
            uc = state.get("user_constraints") or []
            uc.extend(replan["user_constraints"])
            state["user_constraints"] = uc

        # bookkeeping
        state["last_replan_intent"] = replan.get("replan_intent")
        state["last_replan_reason"] = replan.get("replan_reason")

        return json.dumps(state, ensure_ascii=False)

    @staticmethod
    def _wrap_as_message(content: str, response_cost: float = 0.0) -> Any:
        """
        Return an object that behaves like a litellm message:
        has `.content`, `.role`, `.response_cost`, and `.model_dump()`.
        """
        msg = SimpleNamespace(role="assistant", content=content)
        msg.response_cost = response_cost
        msg.model_dump = lambda: {"role": "assistant", "content": content}
        return msg

    @staticmethod
    def _fallback_action(reason: str) -> str:
        """A safe respond action when retries are exhausted or parsing fails."""
        payload = {
            "name": "respond",
            "arguments": {
                "content": (
                    "I'm not able to complete this safely right now. "
                    f"{reason} Could you provide more information, or "
                    "would you like me to transfer you to a human agent?"
                )
            },
        }
        return "Action:\n" + json.dumps(payload, ensure_ascii=False)

    # --------------------------------------------------------------------- #
    # Main pipeline
    # --------------------------------------------------------------------- #
    @classmethod
    def _is_stuck_respond_loop(
        cls, messages: List[Dict[str, Any]], window: int = 4
    ) -> bool:
        """
        Detect whether the agent has emitted >=`window` recent respond
        actions in a row (no successful tool call breaking the run).
        That signals it's stuck politely re-asking for the same thing.
        """
        respond_streak = 0
        # Walk assistant turns in reverse
        for m in reversed(messages):
            if m.get("role") != "assistant":
                continue
            content = m.get("content", "")
            if not isinstance(content, str) or "Action:" not in content:
                continue
            try:
                action = json.loads(content.split("Action:")[-1].strip())
            except (json.JSONDecodeError, TypeError):
                break
            if not isinstance(action, dict):
                break
            if action.get("name") == "respond":
                respond_streak += 1
                if respond_streak >= window:
                    return True
            else:
                # A tool call broke the streak
                return False
        return False

    def orchestrate(self, messages: List[Dict[str, Any]]) -> Any:
        # Reset per-turn cost accumulator
        self._call_cost = 0.0

        turn = len([m for m in messages if m.get("role") == "assistant"]) + 1
        print()
        print(_c("bold", self._divider("═")))
        print(_c("bold", f"  DBSVR  •  turn {turn}  •  {len(messages)} msgs in trajectory"))
        print(_c("bold", self._divider("═")))

        # ---------------------------------------------------------- #
        # 0. Stuck-loop short-circuit  (throttled)                    #
        # ---------------------------------------------------------- #
        # The detector fires at most once every COOLDOWN turns so that
        # its own concession message can't perpetuate the streak. After
        # firing it counts up and the message gets shorter / more
        # terminal each time, hoping the user simulator issues ###STOP###.
        COOLDOWN = 3
        cooled_down = (turn - self._last_stuck_turn) >= COOLDOWN
        if cooled_down and self._is_stuck_respond_loop(messages, window=4):
            self._last_stuck_turn = turn
            self._stuck_fire_count += 1
            print()
            self._log(
                f"▶ [✋] Stuck-loop detected (firing #{self._stuck_fire_count}). "
                "Emitting concession.",
                "red",
            )
            if self._stuck_fire_count == 1:
                msg = (
                    "I understand you'd prefer not to share that detail or "
                    "the verification keeps failing. I can't safely complete "
                    "this request as-is. Would you like me to transfer you "
                    "to a human agent, or is there something else I can do?"
                )
            elif self._stuck_fire_count == 2:
                msg = (
                    "I'm unable to complete this request. Please contact a "
                    "human agent for further assistance. Goodbye."
                )
            else:
                msg = "Goodbye."
            payload = {"name": "respond", "arguments": {"content": msg}}
            final_str = "Action:\n" + json.dumps(payload, ensure_ascii=False)
            self._log(f"  cost       : ${self._call_cost:.6f}", "grey")
            print(_c("bold", self._divider()))
            return self._wrap_as_message(final_str, response_cost=self._call_cost)


        # Format the conversation history once.
        main_traj = self._format_main_traj(messages)
        recent = self._recent_traj(messages, max_lines=12)

        # Context block injected into Simulator / Verifier / Replanner so
        # they can see the most recent turns and avoid re-proposing actions
        # that just happened.
        def _ctx() -> str:
            return (
                self._tools_block()
                + "\n\n## Recent conversation (most recent first below)\n"
                + recent
            )

        # ---------------------------------------------------------- #
        # 1. State Builder                                            #
        # ---------------------------------------------------------- #
        print()
        self._log("▶ [1/5] StateBuilder", "cyan")
        if self.verbose:
            self._log("  trajectory ↓", "grey")
            for ln in main_traj.splitlines():
                print(f"    {_c('grey', ln)}")
        sb_extra = (
            "## Domain policy (for risk + intent classification)\n"
            f"{self.wiki}"
        )
        state_json = self.run_agent(
            user_input=main_traj,
            prompt_file="StateBuilder.yaml",
            extra_system=sb_extra,
        )
        # Print key fields so progress is readable without the full JSON blob
        state_obj = self._safe_json_loads(state_json, {})
        if isinstance(state_obj, dict):
            self._log(
                f"  goal       : {state_obj.get('user_goal', '?')}", "grey"
            )
            self._log(
                f"  subgoal    : {state_obj.get('current_subgoal', '?')}", "grey"
            )
            self._log(
                f"  risk       : {state_obj.get('risk_level', '?')}  |  "
                f"intent: {state_obj.get('intent_type', '?')}", "grey"
            )
            known = state_obj.get("known_facts") or {}
            if isinstance(known, dict) and known:
                # Compact one-line view of known facts (truncate long values)
                kf_compact = ", ".join(
                    f"{k}={(str(v)[:32] + '…') if len(str(v)) > 33 else v}"
                    for k, v in known.items()
                )
                self._log(f"  known      : {kf_compact}", "grey")
            else:
                self._log("  known      : (none)", "grey")
            missing = state_obj.get("missing_facts") or []
            self._log(
                f"  missing    : {missing if missing else '(none)'}", "grey"
            )
            actions = state_obj.get("actions_taken") or []
            if actions:
                last = actions[-1] if isinstance(actions[-1], dict) else {}
                self._log(
                    f"  actions    : {len(actions)} taken  "
                    f"(last: {last.get('tool', '?')} -> "
                    f"{str(last.get('result_summary', ''))[:60]})",
                    "grey",
                )
            else:
                self._log("  actions    : (none yet)", "grey")
        if self.verbose:
            self._log_json("  full state ↓", "grey", state_json)

        # ---------------------------------------------------------- #
        # 2. Budget Scope                                             #
        # ---------------------------------------------------------- #
        print()
        self._log("▶ [2/5] BudgetScope", "cyan")
        budget_json = self.run_agent(
            user_input=state_json,
            prompt_file="BudgetScope.yaml",
        )
        budget_obj = self._safe_json_loads(budget_json, {"budget": "HIGH"})
        budget = str(budget_obj.get("budget", "HIGH")).upper()
        budget_colour = "green" if budget == "LOW" else "yellow"
        self._log(
            f"  budget     : {budget}  —  {budget_obj.get('reason', '')}",
            budget_colour,
        )

        # ---------------------------------------------------------- #
        # 3a. LOW budget                                              #
        # ---------------------------------------------------------- #
        if budget == "LOW":
            print()
            self._log("▶ [3/5] FinalOutput  (LOW path — skipping Simulator/Verifier)", "green")
            final_input = f"State:\n{state_json}\n\nBudget: LOW\n"
            final_str = self.run_agent(
                user_input=final_input,
                prompt_file="FinalOutput.yaml",
                extra_system=self._tools_block(),
            )
            action_preview = final_str.split("Action:")[-1].strip()[:120]
            self._log(f"  action     : {action_preview}", "green")
            self._log(f"  cost       : ${self._call_cost:.6f}", "grey")
            print(_c("bold", self._divider()))
            return self._wrap_as_message(final_str, response_cost=self._call_cost)

        # ---------------------------------------------------------- #
        # 3b. HIGH budget — simulate / verify / replan loop          #
        # ---------------------------------------------------------- #
        sim_json = ""
        ver_json = ""
        for attempt in range(self.MAX_REPLAN_RETRIES):
            attempt_label = f"attempt {attempt + 1}/{self.MAX_REPLAN_RETRIES}"
            print()
            self._log(
                f"▶ [3/5] Simulator  ({attempt_label})", "cyan"
            )

            sim_user = f"State:\n{state_json}"
            sim_json = self.run_agent(
                user_input=sim_user,
                prompt_file="Simulator.yaml",
                extra_system=_ctx(),
            )
            sim_obj = self._safe_json_loads(sim_json, {})
            if isinstance(sim_obj, dict):
                ca = sim_obj.get("candidate_action", {})
                self._log(
                    f"  tool       : {ca.get('tool_name') or '(respond to user)'}",
                    "grey",
                )
                if ca.get("arguments"):
                    self._log(f"  args       : {ca.get('arguments')}", "grey")
                if ca.get("draft_message"):
                    self._log(f"  draft msg  : {ca.get('draft_message')}", "grey")
                self._log(
                    f"  confidence : {sim_obj.get('confidence', '?')}  |  "
                    f"outcome: {sim_obj.get('expected_outcome', '?')[:80]}",
                    "grey",
                )

            # Verifier
            print()
            self._log(f"▶ [4/5] Verifier  ({attempt_label})", "cyan")
            ver_user = (
                f"State:\n{state_json}\n\n"
                f"Proposed Action (from Simulator):\n{sim_json}"
            )
            ver_json = self.run_agent(
                user_input=ver_user,
                prompt_file="Verifier.yaml",
                extra_system=_ctx(),
            )
            ver = self._safe_json_loads(
                ver_json,
                {"accepted": False, "reason": "verifier returned invalid JSON"},
            )
            accepted = bool(ver.get("accepted"))
            v_colour = "green" if accepted else "red"
            v_icon = "✔ ACCEPTED" if accepted else "✘ REJECTED"
            self._log(
                f"  verdict    : {v_icon}  —  {ver.get('reason', '')}",
                v_colour,
            )

            if accepted:
                print()
                self._log("▶ [5/5] FinalOutput  (verified action)", "green")
                final_input = (
                    f"State:\n{state_json}\n\n"
                    f"Budget: HIGH\n\n"
                    f"Simulator Output:\n{sim_json}\n\n"
                    f"Verifier Output:\n{ver_json}"
                )
                final_str = self.run_agent(
                    user_input=final_input,
                    prompt_file="FinalOutput.yaml",
                    extra_system=self._tools_block(),
                )
                action_preview = final_str.split("Action:")[-1].strip()[:120]
                self._log(f"  action     : {action_preview}", "green")
                self._log(f"  cost       : ${self._call_cost:.6f}", "grey")
                print(_c("bold", self._divider()))
                return self._wrap_as_message(
                    final_str, response_cost=self._call_cost
                )

            # Replanner
            print()
            self._log(f"▶ [↺] Replanner  ({attempt_label})", "yellow")
            replan_user = (
                f"State:\n{state_json}\n\n"
                f"Failed Action:\n{sim_json}\n\n"
                f"Verifier Rejection:\n{ver_json}"
            )
            replan_json = self.run_agent(
                user_input=replan_user,
                prompt_file="Replan.yaml",
                extra_system=_ctx(),
            )
            rp = self._safe_json_loads(replan_json, {})
            if isinstance(rp, dict):
                self._log(
                    f"  intent     : {rp.get('replan_intent', '?')}  —  "
                    f"{rp.get('replan_reason', '?')}",
                    "yellow",
                )
                if rp.get("revised_subgoal"):
                    self._log(
                        f"  new subgoal: {rp.get('revised_subgoal')}", "yellow"
                    )

            state_json = self._merge_state(state_json, replan_json)

        # ---------------------------------------------------------- #
        # 3c. Retries exhausted                                       #
        # ---------------------------------------------------------- #
        print()
        self._log(
            f"▶ [5/5] FinalOutput  (FALLBACK — retries exhausted after "
            f"{self.MAX_REPLAN_RETRIES} attempts)",
            "red",
        )
        self._log(f"  cost       : ${self._call_cost:.6f}", "grey")
        print(_c("bold", self._divider()))
        return self._wrap_as_message(
            self._fallback_action(
                "I tried a few approaches but couldn't verify a safe next step."
            ),
            response_cost=self._call_cost,
        )