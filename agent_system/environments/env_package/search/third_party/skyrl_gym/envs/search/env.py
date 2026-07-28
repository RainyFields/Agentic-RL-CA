from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import compute_score
from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import SearchToolGroup
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig


class SearchEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Based on Verl + Search-R1 integration
    """

    def __init__(self, env_config: DictConfig):
        super().__init__()
        # Initialize the tools
        # name is hardcoded to "SearchToolGroup", with tool name "search"
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Agentic-RL-CA Phase 2 (RQ3): B1 privileged answer-exposure step reward.
        # OFF by default ('none') — stock behavior is untouched for every other arm.
        self.step_reward_mode = str(env_config.get("step_reward_mode", "none"))
        self.step_reward_w = float(env_config.get("step_reward_w", 0.0))

        # ASearcher-consistent observation cap (8B protocol): the whole search result set
        # (the content of one <information> block) is truncated to this many characters,
        # mirroring ASearcher's 5k-char cap on <search> observations. 0 = off (stock).
        self.info_char_cap = int(env_config.get("info_char_cap", 0))

    def reset(self, extras: Dict[str, Any] = {}) -> None:
        assert "ground_truth" in extras, "ground_truth is required in extras field"
        self.ground_truth = extras["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 3
        self._b1_given = False  # first-hit-only latch (one step reward per trajectory)

        self.data_source = extras.get("data_source", "unknown")

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []
        self.done = False
        self.turns = 0


    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps for Search tasks
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _postprocess_action(self, action: str) -> str:
        if "</search>" in action:
            return action.split("</search>")[0] + "</search>"
        elif "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        else:
            return action

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)
        if len(tool_output) > 0:
            if self.info_char_cap > 0 and len(tool_output) > self.info_char_cap:
                tool_output = tool_output[: self.info_char_cap]
            return "\n<information>" + tool_output + "</information>\n"
        else:
            return None

    # ---- Agentic-RL-CA Phase 2b: exact state snapshot/restore (search state is fully
    # driver-local and deterministic given the query, so snapshots are exact). Shared
    # infrastructure for CARL tree rollouts AND the Phase-3b credit-alignment diagnostic
    # (prefix resume). ----
    def snapshot_state(self) -> Dict[str, Any]:
        from copy import deepcopy
        return {
            "ground_truth": deepcopy(self.ground_truth),
            "max_turns": self.max_turns,
            "data_source": self.data_source,
            "chat_history": deepcopy(self.chat_history),
            "done": self.done,
            "turns": self.turns,
            "_b1_given": getattr(self, "_b1_given", False),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        from copy import deepcopy
        self.ground_truth = deepcopy(state["ground_truth"])
        self.max_turns = state["max_turns"]
        self.data_source = state["data_source"]
        self.chat_history = deepcopy(state["chat_history"])
        self.done = state["done"]
        self.turns = state["turns"]
        self._b1_given = state.get("_b1_given", False)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        # action = self._postprocess_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        if not self.done:
            done = self._is_done(action)
            self.done = done
        else:
            done = True

        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={"data_source": self.data_source, "tool_calling": False}, postprocessed_action=action
            )

        try:
            query = self._parse_action(action)
            observation = self._execute_tool("SearchToolGroup", "search", query)
        except Exception as e:
            error = str(e)
            observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            print(f"!!(Warning) an error when calling tools: {error}")
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_calling": True,
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
            "data_source": self.data_source,
        }

        if self.step_reward_mode == "b1":
            from credit_assignment.step_rewards import compute_b1_step_reward

            bonus, hit, self._b1_given = compute_b1_step_reward(
                observation if isinstance(observation, str) else None,
                self.ground_truth,
                self._b1_given,
                self.step_reward_w,
            )
            reward += bonus
            info["b1_hit"] = hit
            info["b1_step_reward"] = bonus

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )