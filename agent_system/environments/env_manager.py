# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.prompts.alfworld_transcript import build_transcript_prompt
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask

    # ---- Agentic-RL-CA Phase 2b: exact snapshot/restore of the full driver-local state
    # (env chat_history/turns + SearchMemory prefix + task). Shared infrastructure for
    # CARL rollout trees and the Phase-3b credit-alignment diagnostic (prefix resume). ----
    def snapshot(self, indices: List[int] = None) -> List[Dict]:
        from copy import deepcopy
        idxs = list(range(self.memory.batch_size)) if indices is None else list(indices)
        env_states = self.envs.get_states(idxs)
        return [
            {
                "env": es,
                "memory": deepcopy(self.memory._data[i]),
                "task": self.tasks[i],
            }
            for es, i in zip(env_states, idxs)
        ]

    def restore_batch(self, snapshots: List[Dict], indices: List[int] = None) -> Dict[str, Any]:
        """Restore snapshots into env slots `indices`; returns the policy-visible
        observation dict at the restored states (same structure as reset()/step())."""
        from copy import deepcopy
        idxs = list(range(len(snapshots))) if indices is None else list(indices)
        assert len(snapshots) == len(idxs)
        self.envs.set_states([s["env"] for s in snapshots], idxs)
        for s, i in zip(snapshots, idxs):
            self.memory._data[i] = deepcopy(s["memory"])
            self.tasks[i] = s["task"]
        anchors = []
        for i in idxs:
            if len(self.memory._data[i]) > 0:
                anchors.append(self.memory._data[i][-1].get("information", self.tasks[i]))
            else:
                anchors.append(self.tasks[i])
        return {
            "text": self.rebuild_text_obs(idxs),
            "image": None,
            "anchor": anchors,
        }

    def rebuild_text_obs(self, indices: List[int]) -> List[str]:
        """Rebuild the templated per-turn prompt for the given env slots from the current
        (possibly just-restored) memory + task state. Mirrors build_text_obs() exactly
        (all limits read from config — no literal turn counts)."""
        history_length = self.config.env.history_length
        out = []
        for i in indices:
            n_steps = len(self.memory._data[i])
            if n_steps == 0 or history_length <= 0:
                out.append(SEARCH_TEMPLATE_NO_HIS.format(task_description=self.tasks[i]))
            else:
                recent = self.memory._data[i][-history_length:]
                start_idx = n_steps - len(recent)
                lines = [
                    f"Step {start_idx + j + 1}:{rec['search']} {rec['information']}\n"
                    for j, rec in enumerate(recent)
                ]
                out.append(
                    SEARCH_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context="\n".join(lines),
                        step_count=n_steps,
                    )
                )
        return out


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        # ReAct responses are "Thought: ...\nAction: <act>" -> use the lenient parser (strips Action:, matches admissible)
        _weak = bool(self.config.env.get('use_admissible_action_prompt', False)) or \
                bool(self.config.env.get('use_react_prompt', False)) or \
                bool(self.config.env.get('use_react_transcript', False))
        actions, valids, parse_statuses = self.projection_f(
            text_actions, self.envs.get_admissible_commands,
            weak_prompt=_weak,
            match_mode=self.config.env.get('admissible_action_match_mode', 'weak_text_match'),
            replace_invalid_with_random=self.config.env.get('replace_invalid_with_random_admissible', False),
        )
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'response': text_actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid + parse_status to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['parse_status'] = parse_statuses[i]

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        weak = bool(self.config.env.get('use_admissible_action_prompt', False))
        react = bool(self.config.env.get('use_react_prompt', False))   # SP6 ReAct track (matches BC pi_base)
        transcript = bool(self.config.env.get('use_react_transcript', False))  # SP6 full-history + one-shot
        for i in range(len(text_obs)):
            if transcript:
                tw = int(self.config.env.get('transcript_window', 0) or 0)
                recs = self.memory[i]
                if not recs:
                    obs = build_transcript_prompt(text_obs[i], [], max_steps=tw)
                else:
                    steps = []
                    for k in range(len(recs)):
                        nxt = recs[k + 1]['text_obs'] if k + 1 < len(recs) else text_obs[i]
                        steps.append((recs[k].get('response', recs[k]['action']), nxt))
                    obs = build_transcript_prompt(recs[0]['text_obs'], steps, max_steps=tw)
                postprocess_text_obs.append(obs)
                continue
            # exclude 'help' in admissible_actions[i]
            acts = [s for s in admissible_actions[i] if s != 'help']
            if weak:
                reformatted_admissible_actions = "\n".join(f"- {s}" for s in acts)
            else:
                reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in acts)

            if react:
                # ReAct prompt — actions are in the preamble, not listed per turn; no admissible list.
                if init or self.config.env.history_length <= 0:
                    obs = ALFWORLD_TEMPLATE_REACT_NO_HIS.format(current_observation=text_obs[i])
                else:
                    obs = ALFWORLD_TEMPLATE_REACT.format(
                        task_description=self.tasks[i],
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
                continue

            if init or self.config.env.history_length <= 0:
                tmpl = ALFWORLD_TEMPLATE_WEAK_NO_HIS if weak else ALFWORLD_TEMPLATE_NO_HIS
                obs = tmpl.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                tmpl = ALFWORLD_TEMPLATE_WEAK if weak else ALFWORLD_TEMPLATE
                obs = tmpl.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SciWorldEnvironmentManager(EnvironmentManagerBase):
    """Rung-4 ScienceWorld manager: ReAct grammar + three-tier K-recent truncation
    (design doc docs/reports/2026-07-28_sciworld_orm_prm_design/design.md). The
    truncation rule is part of the MDP: identical across arms and at eval."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self._tokenizer = None
        self._fixed_tokens = None
        self.trunc_counts = {"tier2_turns": 0, "tier3_drops": 0, "emergency": 0, "prompts": 0}
        super().__init__(envs, projection_f, config)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            path = self.config.env.sciworld.get("tokenizer_path", None) or \
                self.config.actor_rollout_ref.model.path
            self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        return self._tokenizer

    def _ntok(self, s: str) -> int:
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        n = len(text_obs)
        self.memory.reset(batch_size=n)
        self.tasks = [info["task_description"] for info in infos]
        self.pre_text_obs = text_obs
        fixed = [SCIWORLD_TEMPLATE.format(task_description=t, history="",
                                          turn=0, current_observation="") for t in self.tasks]
        self._fixed_tokens = [self._ntok(f) + 16 for f in fixed]  # +16: turn-number digits etc.

        full_text_obs = self.build_text_obs(text_obs)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids, parse_statuses = self.projection_f(text_actions)
        thoughts = [self._extract_thought(r) for r in text_actions]
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)

        turn_nos = [len(self.memory[i]) + 1 for i in range(len(actions))]
        hist_full = [SCIWORLD_HIST_FULL.format(thought=thoughts[i], action=actions[i],
                                               turn=turn_nos[i], obs=self.pre_text_obs[i])
                     for i in range(len(actions))]
        hist_act = [SCIWORLD_HIST_ACTION_ONLY.format(turn=turn_nos[i], action=actions[i])
                    for i in range(len(actions))]
        self.memory.store({
            'text_obs': self.pre_text_obs, 'action': actions,
            'hist_full': hist_full, 'hist_act': hist_act,
            'ntok_full': [self._ntok(s) for s in hist_full],
            'ntok_act': [self._ntok(s) for s in hist_act],
        })
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(bool(valids[i]) and info.get('env_action_valid', True))
            info['parse_status'] = parse_statuses[i]

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    @staticmethod
    def _extract_thought(response: str) -> str:
        if not isinstance(response, str):
            return ""
        m = response.split("Action:")[0]
        m = m.replace("Thought:", "", 1).strip()
        return m[:600]

    def build_text_obs(self, text_obs: List[str]) -> List[str]:
        K = int(self.config.env.sciworld.get("recent_k", 20))
        margin = int(self.config.env.sciworld.get("prompt_token_margin", 768))
        budget = int(self.config.data.max_prompt_length) - margin

        out = []
        for i in range(len(text_obs)):
            recs = self.memory[i]
            turn = len(recs) + 1
            avail = budget - self._fixed_tokens[i] - self._ntok(text_obs[i])
            recent, older = recs[-K:], recs[:-K]

            older_toks = sum(r['ntok_act'] for r in older)
            recent_toks = sum(r['ntok_full'] for r in recent)
            self.trunc_counts["prompts"] += 1
            self.trunc_counts["tier2_turns"] += len(older)

            drop = 0
            while older_toks + recent_toks > avail and drop < len(older):
                older_toks -= older[drop]['ntok_act']  # tier 3: drop oldest action lines
                drop += 1
            if drop:
                self.trunc_counts["tier3_drops"] += drop
            older = older[drop:]

            demote = 0  # emergency: last K full turns alone overflow -> demote oldest to action-only
            while older_toks + recent_toks > avail and demote < len(recent) - 1:
                recent_toks += recent[demote]['ntok_act'] - recent[demote]['ntok_full']
                older_toks += 0
                demote += 1
            if demote:
                self.trunc_counts["emergency"] += demote
                print(f"[sciworld-trunc] EMERGENCY demote={demote} env={i} turn={turn}")

            parts = [r['hist_act'] for r in older]
            parts += [recent[j]['hist_act'] if j < demote else recent[j]['hist_full']
                      for j in range(len(recent))]
            history = "".join(parts)
            if history:
                history += "\n"
            out.append(SCIWORLD_TEMPLATE.format(
                task_description=self.tasks[i], history=history,
                turn=turn, current_observation=text_obs[i]))
        return out

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][i]['active_masks']:
                info = total_infos[batch_idx][i]
                won = float(info['won'])
                success['success_rate'].append(won)
                success[f"{info['family']}_success_rate"].append(won)
                success['sw_score'].append(float(info.get('progress', 0.0)))
                seq_total = info.get('seq_total', 0)
                success['sw_seq_progress'].append(
                    float(info.get('seq_done', 0)) / seq_total if seq_total else 0.0)
                success['sw_focus_death'].append(float(info.get('focus_death', False)))
                success['sw_cap_hit'].append(float(info.get('cap_hit', False)))
                return

    def success_evaluator(self, *args, **kwargs):
        success = super().success_evaluator(*args, **kwargs)
        c = self.trunc_counts
        if c["prompts"]:
            success['sw_trunc_tier2_turns_per_prompt'] = np.array([c["tier2_turns"] / c["prompts"]])
            success['sw_trunc_tier3_drops_per_prompt'] = np.array([c["tier3_drops"] / c["prompts"]])
            success['sw_trunc_emergency_per_prompt'] = np.array([c["emergency"] / c["prompts"]])
        self.trunc_counts = {"tier2_turns": 0, "tier3_drops": 0, "emergency": 0, "prompts": 0}
        return success


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        env_kwargs = {
            'reward_mode': config.env.sciworld.reward_mode,
            'env_step_limit': config.env.sciworld.env_step_limit,
        }
        _envs = build_sciworld_envs(config.env.seed, config.data.train_batch_size, group_n,
                                    resources_per_worker, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sciworld_envs(config.env.seed + 1000, config.data.val_batch_size, 1,
                                        resources_per_worker, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(sciworld_projection)
        envs = SciWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = SciWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)