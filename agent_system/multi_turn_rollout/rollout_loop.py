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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        # Agentic-RL-CA: pre-truncation prompt length — any prompt-side truncation event is a
        # protocol bug (plan: truncation-event rate must be <0.1%), so it must be measurable.
        prompt_pretrunc_len = len(raw_prompt_ids)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'prompt_pretrunc_len': prompt_pretrunc_len,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def _turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            obs: Dict,
            uid_batch: np.ndarray,
            traj_uid: np.ndarray,
            turn_offsets: np.ndarray = None,
            pre_gen_hook=None,
            post_step_hook=None,
            max_steps_override: int = None,
            ):
        """Shared agent-environment turn loop (Agentic-RL-CA Phase 2b.2 refactor).

        Runs the generate→step loop for `len(gen_batch.batch)` env slots starting from
        `obs` (which may come from envs.reset() — vanilla — or envs.restore_batch() —
        CARL phase-2 resumes). Behavior is byte-identical to the pre-refactor vanilla
        loop when turn_offsets is None and no hooks are given.

        Parameters beyond the vanilla ones:
            obs: initial policy-visible observation dict for the slots being rolled.
            uid_batch / traj_uid: caller-owned identity arrays (length = batch size).
            turn_offsets: per-slot starting global depth; row turn_index = offset + step
                (None → zeros). Episode lengths start at the offset so they stay global.
            pre_gen_hook(step, active_masks, batch): called after preprocess_batch,
                before the generation pop — CARL snapshots env state / hashes the
                policy-visible prompt here.
            post_step_hook(step, active_masks, dones, batch): called after env.step once
                all per-row fields are attached, before rows are split — may attach
                additional non_tensor fields (must be UNCONDITIONAL per-row: ragged
                keys break collate_fn, see b1_hit note below).

        Returns (total_batch_list, episode_rewards, episode_lengths, tool_callings,
                 total_infos, rollout_records).
        """
        batch_size = len(gen_batch.batch)
        if turn_offsets is None:
            turn_offsets = np.zeros(batch_size, dtype=np.int64)

        is_done = np.zeros(batch_size, dtype=bool)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = turn_offsets.astype(np.float32).copy()
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # SP3.1: rollout debugging + early-stop bookkeeping
        es_invalid = self.config.env.get('early_stop_invalid_count', None)
        es_repeated = self.config.env.get('early_stop_repeated_action_count', None)
        es_nochange = self.config.env.get('early_stop_no_state_change_count', None)
        invalid_count = np.zeros(batch_size, dtype=np.int64)    # cumulative invalid actions
        repeated_count = np.zeros(batch_size, dtype=np.int64)   # consecutive repeated actions
        nochange_count = np.zeros(batch_size, dtype=np.int64)   # consecutive no-state-change
        early_stopped = np.zeros(batch_size, dtype=bool)
        early_stop_reason = np.array([""] * batch_size, dtype=object)
        prev_action = [None] * batch_size
        prev_obs_text = [None] * batch_size
        rollout_records = []   # SP3.1: reliable per-turn debug log (written to JSONL at loop end)
        # Lean async plan P2 (baseline instrumentation, +env.rollout_profiling=true):
        # count generated tokens on active vs done ("zombie") rows per cycle.
        _profiling = bool(self.config.env.get('rollout_profiling', False))
        _tok_active = 0
        _tok_zombie = 0
        # Sync partial rollout: per-cycle turn budget (global horizon is still enforced by
        # the env's own max_turns, which persists across snapshot/restore).
        n_loop_steps = max_steps_override if max_steps_override is not None else self.config.env.max_steps
        # Trajectory collection loop
        for _step in range(n_loop_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            if pre_gen_hook is not None:
                pre_gen_hook(_step, active_masks, batch)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)

            # Sync partial rollout: per-SEGMENT behavior-policy provenance. Every row is
            # stamped with the global_step of the policy that generated it (set by the
            # trainer via gen_batch.meta_info['policy_version']; -1 when absent, e.g. val).
            # A trajectory resumed across cycles carries rows with different versions.
            batch.non_tensor_batch['policy_version'] = np.full(
                batch_size, int(gen_batch.meta_info.get('policy_version', -1)), dtype=object)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Agentic-RL-CA: per-ROW tool-calling flag (B1-shuffle eligibility) and raw
            # retrieval-hit exposure signal (b1 arms; False elsewhere). UNCONDITIONAL on
            # every step: per-step gating made the key ragged across turns (steps where all
            # envs terminate carry no search metadata) and collate_fn asserted
            # "key b1_hit length 87 != batch size 109".
            batch.non_tensor_batch['tool_calling'] = np.array(
                [bool(info.get('tool_calling', False)) for info in infos], dtype=object)
            batch.non_tensor_batch['b1_hit'] = np.array(
                [bool(info.get('b1_hit', False)) for info in infos], dtype=object)
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            # SP3: per-turn step index within the trajectory (for cross-turn GAE boundary
            # detection). GLOBAL depth: resumed slots (CARL phase 2) start at their fork
            # depth, vanilla slots at 0.
            batch.non_tensor_batch['turn_index'] = turn_offsets + _step

            # SP3.1: per-turn debugging fields + loop counters + early stop (active envs only)
            rewards_np = torch_to_numpy(rewards)
            dones_np = np.asarray(dones, dtype=bool)
            is_valid_arr = batch.non_tensor_batch['is_action_valid']
            parse_status = np.array([info.get('parse_status', 'unknown') for info in infos], dtype=object)
            next_text = next_obs.get('text') if isinstance(next_obs, dict) else None
            cur_text = obs.get('text') if isinstance(obs, dict) else None
            resp_len0 = batch.batch['responses'].shape[1]
            resp_tok0 = torch_to_numpy(batch.batch['attention_mask'][:, -resp_len0:].sum(-1)).astype(np.int64)
            for i in range(batch_size):
                if not active_masks[i]:
                    continue
                if not bool(is_valid_arr[i]):
                    invalid_count[i] += 1
                if prev_action[i] is not None and text_actions[i] == prev_action[i]:
                    repeated_count[i] += 1
                else:
                    repeated_count[i] = 0
                prev_action[i] = text_actions[i]
                cur_obs = next_text[i] if next_text is not None else None
                if prev_obs_text[i] is not None and cur_obs is not None and cur_obs == prev_obs_text[i]:
                    nochange_count[i] += 1
                else:
                    nochange_count[i] = 0
                prev_obs_text[i] = cur_obs
                if not bool(dones_np[i]):  # only rollout-truncate envs the env hasn't already ended
                    reason = ""
                    if es_invalid is not None and invalid_count[i] >= es_invalid:
                        reason = f"invalid_count>={es_invalid}"
                    elif es_repeated is not None and repeated_count[i] >= es_repeated:
                        reason = f"repeated_action>={es_repeated}"
                    elif es_nochange is not None and nochange_count[i] >= es_nochange:
                        reason = f"no_state_change>={es_nochange}"
                    if reason:
                        early_stopped[i] = True
                        early_stop_reason[i] = reason
                rollout_records.append({
                    "traj_uid": str(traj_uid[i]), "turn_index": int(turn_offsets[i] + _step),
                    "uid": str(uid_batch[i]),
                    "data_source": str(batch.non_tensor_batch['data_source'][i]) if 'data_source' in batch.non_tensor_batch else "",
                    "prompt_pretrunc_len": int(batch.non_tensor_batch['prompt_pretrunc_len'][i]) if 'prompt_pretrunc_len' in batch.non_tensor_batch else -1,
                    "parse_status": str(parse_status[i]), "is_action_valid": bool(is_valid_arr[i]),
                    "env_reward": float(rewards_np[i]), "env_done": bool(dones_np[i]),
                    "env_won": bool(bool(dones_np[i]) and float(rewards_np[i]) > 0),
                    "invalid_count_so_far": int(invalid_count[i]),
                    "repeated_count_so_far": int(repeated_count[i]),
                    "response_token_count": int(resp_tok0[i]),
                    "early_stopped": bool(early_stopped[i]), "early_stop_reason": str(early_stop_reason[i]),
                    "observation": (cur_text[i] if cur_text is not None else ""),
                    "raw_model_response": text_actions[i],
                    # rung-4 sciworld gate fields (guarded .get: empty/False for other envs)
                    "sw_task": str(infos[i].get("task", "")),
                    "sw_variation": int(infos[i].get("variation", -1)),
                    "sw_progress": float(infos[i].get("progress", 0.0)),
                    "sw_won": bool(infos[i].get("won", False)),
                    "sw_focus_death": bool(infos[i].get("focus_death", False)),
                    "sw_cap_hit": bool(infos[i].get("cap_hit", False)),
                    "sw_env_crash": bool(infos[i].get("env_crash", False)),
                    # Agentic-RL-CA: RAW retrieved information for THIS turn's action (the
                    # anchor of next_obs, un-templated) — lets the credit-alignment
                    # diagnostic rebuild exact prefix states from dumps alone.
                    "information": (
                        str(next_obs.get('anchor')[i])
                        if isinstance(next_obs, dict) and next_obs.get('anchor') is not None
                        else ""
                    ),
                })
            resp_len = batch.batch['responses'].shape[1]
            resp_tok = batch.batch['attention_mask'][:, -resp_len:].sum(-1)
            if _profiling:
                _rt = torch_to_numpy(resp_tok)
                _tok_active += int(_rt[active_masks].sum())
                _tok_zombie += int(_rt[~active_masks].sum())
            batch.non_tensor_batch['parse_status'] = parse_status
            batch.non_tensor_batch['env_reward'] = torch_to_numpy(rewards, is_object=True)   # raw env reward
            batch.non_tensor_batch['env_done'] = np.array(dones_np, dtype=object)
            batch.non_tensor_batch['env_won'] = np.array(dones_np & (rewards_np > 0), dtype=object)
            batch.non_tensor_batch['invalid_count_so_far'] = invalid_count.copy()
            batch.non_tensor_batch['repeated_count_so_far'] = repeated_count.copy()
            batch.non_tensor_batch['response_token_count'] = torch_to_numpy(resp_tok).astype(np.int64)
            batch.non_tensor_batch['early_stopped'] = np.array(early_stopped, dtype=object)
            batch.non_tensor_batch['early_stop_reason'] = early_stop_reason.copy()

            if post_step_hook is not None:
                post_step_hook(_step, active_masks, dones_np, batch)

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states (env terminal OR SP3.1 early-stop rollout-truncation)
            is_done = np.logical_or(is_done, dones)
            is_done = np.logical_or(is_done, early_stopped)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        if _profiling:
            _tot = _tok_active + _tok_zombie
            print(f"[sync_rollout_profile] generated tokens: active={_tok_active} "
                  f"zombie={_tok_zombie} zombie_ratio="
                  f"{(_tok_zombie / _tot if _tot else 0.0):.4f}", flush=True)

        return (total_batch_list, episode_rewards, episode_lengths, tool_callings,
                total_infos, rollout_records)

    def _dump_rollout_records(self, rollout_records):
        """SP3.1: write the per-turn rollout debug log (reliable; bypasses non_tensor
        propagation). SP4 HISR STAGE-1: the same per-turn JSONL (obs holds the admissible
        list, raw_model_response is the action, env_won=success) is the HISR collector's
        input — enable it via HISR_COLLECT=1."""
        import os as _os, json as _json
        _hisr_collect = _os.environ.get("HISR_COLLECT") == "1"
        if (_os.environ.get("DUMP_TRAIN_SAMPLE") == "1" or _hisr_collect) and rollout_records:
            _dir = _os.environ.get("HISR_COLLECT_PATH") if _hisr_collect else _os.environ.get("DUMP_TRAIN_PATH", "/tmp/ppo_true_dump")
            _dir = _dir or "/tmp/ppo_true_dump"
            _p = _os.path.join(_dir, "rollout_log.jsonl")
            _os.makedirs(_os.path.dirname(_p), exist_ok=True)
            with open(_p, "a") as _fh:
                for _r in rollout_records:
                    _fh.write(_json.dumps(_r) + "\n")

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances

        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        (total_batch_list, episode_rewards, episode_lengths, tool_callings,
         total_infos, rollout_records) = self._turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            obs=obs,
            uid_batch=uid_batch,
            traj_uid=traj_uid,
        )

        self._dump_rollout_records(rollout_records)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def _policy_prompt_ids(self, obs_text: str):
        """Tokenize an observation into the exact policy-visible prompt ids that
        preprocess_single_sample would produce as raw_prompt_ids (chat template +
        add_generation_prompt + protocol truncation). CARL node identity hashes these."""
        chat = np.array([{"content": obs_text, "role": "user"}])
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        prompt = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_len = self.config.data.max_prompt_length
        if len(ids) > max_len:
            trunc = self.config.data.truncation
            if trunc == "left":
                ids = ids[-max_len:]
            elif trunc == "right":
                ids = ids[:max_len]
            elif trunc == "middle":
                left_half = max_len // 2
                ids = ids[:left_half] + ids[-(max_len - left_half):]
        return ids

    def carl_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ):
        """Agentic-RL-CA Phase 2b.2 — CARL two-phase tree rollout (arXiv 2512.04949;
        constants/deltas locked in docs/methods_note.md).

        Phase 1: n0 stochastic rollouts per prompt, snapshotting the exact driver-local
        state at every visited depth. Phase 2: n_total−n0 resumes per prompt, assigned
        round-robin over the group's snapshot depths, each continuing STOCHASTICALLY from
        the restored state (methods_note: resume temp 0 appears only in the paper's Eq. 5
        preliminary study; the unbiasedness argument requires sampling from pi_theta).

        Every generated turn is one tree edge; rows carry carl_src/carl_dst node ids
        (sha1 of the policy-visible tokenized prompt before/after the action) consumed by
        credit_assignment.core_carl at advantage time. Runtime fidelity counters (soft):
        chain (dst_t == src_{t+1}) and resume (restored prompt == snapshot node).
        """
        from credit_assignment.state_tools import node_id_from_prompt_ids

        ccfg = self.config.algorithm.get('carl', {}) or {}
        n0 = int(ccfg.get('n0', 1))
        n_total = int(ccfg.get('n_total', 16))
        include_root = bool(ccfg.get('include_root', False))
        assert 1 <= n0 < n_total, f"CARL needs 1 <= n0 < n_total, got n0={n0} n_total={n_total}"

        n_prompts = len(gen_batch.batch)
        pool = envs.envs.batch_size  # total env worker slots
        assert pool >= n_prompts * n0, (
            f"CARL phase 1 needs {n_prompts * n0} env slots, pool has {pool} "
            f"(set env.rollout.n >= max(n0, n_total-n0))")

        group_uid = [str(uuid.uuid4()) for _ in range(n_prompts)]
        fidelity = {"chain_mismatch": 0, "resume_mismatch": 0, "merge_mismatch": 0}
        node_ids_seen = {}  # node -> tuple(prompt ids); hard identity audit (soft at runtime)

        def _audit_node(node, ids):
            prev = node_ids_seen.get(node)
            tup = tuple(int(t) for t in ids)
            if prev is None:
                node_ids_seen[node] = tup
            elif prev != tup:
                fidelity["merge_mismatch"] += 1

        class _Hooks:
            """Per-phase closure state for the shared _turn_loop callbacks."""
            def __init__(hself, size, offsets, groups, snap_store, expect_first, resumed):
                hself.size = size
                hself.offsets = offsets          # per-slot global start depth
                hself.groups = groups            # per-slot prompt-group index
                hself.snap_store = snap_store    # list per group (phase 1) or None
                hself.expect = list(expect_first)  # expected src node per slot (or None)
                hself.resumed = resumed

            def pre_gen(hself, _step, active_masks, batch):
                ids_list = batch.non_tensor_batch['raw_prompt_ids']
                act_idx = [i for i in range(hself.size) if active_masks[i]]
                hself.src = [""] * hself.size
                snaps = envs.snapshot(act_idx) if hself.snap_store is not None else None
                for j, i in enumerate(act_idx):
                    node = node_id_from_prompt_ids(ids_list[i])
                    _audit_node(node, ids_list[i])
                    if hself.expect[i] is not None and hself.expect[i] != node:
                        key = "resume_mismatch" if (_step == 0 and hself.resumed) else "chain_mismatch"
                        fidelity[key] += 1
                    hself.src[i] = node
                    if snaps is not None:
                        hself.snap_store[hself.groups[i]].append({
                            "depth": int(hself.offsets[i] + _step),
                            "snap": snaps[j],
                            "node": node,
                        })

            def post_step(hself, _step, active_masks, dones, batch):
                texts = envs.rebuild_text_obs(list(range(hself.size)))
                dst = np.empty(hself.size, dtype=object)
                for i in range(hself.size):
                    if active_masks[i]:
                        ids = self._policy_prompt_ids(texts[i])
                        dst[i] = node_id_from_prompt_ids(ids)
                        _audit_node(dst[i], ids)
                        hself.expect[i] = dst[i]  # chain check next turn
                    else:
                        dst[i] = ""
                batch.non_tensor_batch['carl_src'] = np.array(hself.src, dtype=object)
                batch.non_tensor_batch['carl_dst'] = dst
                batch.non_tensor_batch['carl_resumed'] = np.full(
                    hself.size, hself.resumed, dtype=object)

        # ---------------- phase 1: n0 rollouts per prompt, snapshot every depth --------
        p1_batch = gen_batch.repeat(repeat_times=n0, interleave=True) if n0 > 1 else gen_batch
        obs, _ = envs.reset(kwargs=p1_batch.non_tensor_batch.pop('env_kwargs', None))
        p1_size = n_prompts * n0
        uid1 = np.array([group_uid[i // n0] for i in range(p1_size)], dtype=object)
        tuid1 = np.array([str(uuid.uuid4()) for _ in range(p1_size)], dtype=object)
        snapshots = [[] for _ in range(n_prompts)]
        hooks1 = _Hooks(size=p1_size,
                        offsets=np.zeros(p1_size, dtype=np.int64),
                        groups=[i // n0 for i in range(p1_size)],
                        snap_store=snapshots,
                        expect_first=[None] * p1_size,
                        resumed=False)
        (total_batch_list, episode_rewards, episode_lengths, tool_callings,
         total_infos, rollout_records) = self._turn_loop(
            gen_batch=p1_batch, actor_rollout_wg=actor_rollout_wg, envs=envs, obs=obs,
            uid_batch=uid1, traj_uid=tuid1,
            pre_gen_hook=hooks1.pre_gen, post_step_hook=hooks1.post_step)
        traj_uid_all = list(tuid1)

        # ---------------- phase 2: round-robin resumes over snapshot depths ------------
        from credit_assignment.core_carl import carl_resume_schedule
        n_resume = n_total - n0
        resumes = []  # (group, snapshot-entry)
        for g in range(n_prompts):
            resumes.extend((g, s) for s in carl_resume_schedule(snapshots[g], n_resume, include_root))

        for wave_start in range(0, len(resumes), pool):
            wave = resumes[wave_start:wave_start + pool]
            k = len(wave)
            # size the manager's driver-local state to this wave before restoring
            envs.memory.reset(batch_size=k)
            envs.tasks = [""] * k
            obs2 = envs.restore_batch([s["snap"] for _, s in wave], indices=list(range(k)))
            wave_gen = gen_batch.select_idxs(np.array([g for g, _ in wave], dtype=np.int64))
            if 'env_kwargs' in wave_gen.non_tensor_batch:
                wave_gen.non_tensor_batch.pop('env_kwargs')
            offsets = np.array([s["depth"] for _, s in wave], dtype=np.int64)
            uid2 = np.array([group_uid[g] for g, _ in wave], dtype=object)
            tuid2 = np.array([str(uuid.uuid4()) for _ in range(k)], dtype=object)
            hooks2 = _Hooks(size=k, offsets=offsets,
                            groups=[g for g, _ in wave],
                            snap_store=None,
                            expect_first=[s["node"] for _, s in wave],
                            resumed=True)
            (bl, er, el, tc, ti, rr) = self._turn_loop(
                gen_batch=wave_gen, actor_rollout_wg=actor_rollout_wg, envs=envs, obs=obs2,
                uid_batch=uid2, traj_uid=tuid2, turn_offsets=offsets,
                pre_gen_hook=hooks2.pre_gen, post_step_hook=hooks2.post_step)
            total_batch_list += bl
            episode_rewards = np.concatenate([episode_rewards, er])
            episode_lengths = np.concatenate([episode_lengths, el])
            tool_callings = np.concatenate([tool_callings, tc])
            total_infos += ti
            rollout_records += rr
            traj_uid_all += list(tuid2)

        n_snap = sum(len(s) for s in snapshots)
        print(f"[CARL] prompts={n_prompts} n0={n0} n_total={n_total} snapshots={n_snap} "
              f"resumes={len(resumes)} fidelity={fidelity}", flush=True)

        self._dump_rollout_records(rollout_records)
        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        traj_uid = np.array(traj_uid_all, dtype=object)
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def partial_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            ):
        """Synchronous partial rollout (PoC — Agentic-RL-CA, 2026-07-29 design doc).

        Semantics (simplest-safe):
          - Each trainer cycle runs at most `env.partial_rollout_cycle_turns` turns.
          - Trajectories not terminated at cycle end are snapshotted (env state via
            SearchEnvironmentManager.snapshot + accumulated rows) and RESUMED next cycle
            under the then-current policy (cross-update resume). Nothing is restarted.
          - A trajectory enters training only once terminated, and a uid GROUP is
            released ATOMICALLY (all `env.rollout.n` members terminated) so the GRPO
            group baseline is never computed over a split group.
          - Every row records `policy_version` (trainer global_step at generation).
            NO off-policy correction is applied: old_log_probs are still recomputed at
            train time under the current policy (see design doc, risk R2).
          - Groups whose pending members exceed `env.partial_rollout_max_age` cycles are
            dropped entirely (staleness bound; logged as partial/dropped_stale_groups).

        Returns a finalized DataProto of released trajectories, or None if no group
        completed this cycle (trainer skips the update).
        """
        cfg_env = self.config.env
        n = cfg_env.rollout.n if cfg_env.rollout.n > 0 else 1
        pool = self.config.data.train_batch_size * n
        cycle_turns = int(cfg_env.get('partial_rollout_cycle_turns', cfg_env.max_steps))
        max_age = int(cfg_env.get('partial_rollout_max_age', 4))

        if not hasattr(self, '_pr_groups'):
            self._pr_groups: Dict[str, Dict] = {}   # uid -> group record
            self._pr_cycle = 0
        self._pr_cycle += 1

        # ---- 1. staleness bound: drop whole groups with over-age pending members ----
        dropped_groups = [u for u, g in self._pr_groups.items()
                          if g['pending'] and (self._pr_cycle - g['born']) > max_age]
        for u in dropped_groups:
            del self._pr_groups[u]

        # ---- 2. pack env slots: fresh prompt groups in [0..F), resumed pending in [F..F+R) ----
        pending = [(u, t) for u, g in self._pr_groups.items() for t in list(g['pending'])]
        n_resume = len(pending)
        n_fresh_groups = min(max(0, (pool - n_resume) // n), len(gen_batch))

        parts = []
        env_kwargs = []
        if n_fresh_groups > 0:
            gb_fresh = gen_batch.select_idxs(list(range(n_fresh_groups)))
            gb_fresh = gb_fresh.repeat(repeat_times=n, interleave=True)
            ek = gb_fresh.non_tensor_batch.pop('env_kwargs', None)
            assert ek is not None, "partial rollout: search env requires env_kwargs per prompt"
            env_kwargs = list(ek)
            parts.append(gb_fresh)
        for u, t in pending:
            parts.append(self._pr_groups[u]['pending'][t]['gen_row'])
        if not parts:
            self._partial_metrics = {'partial/released_traj': 0, 'partial/pending_traj': 0}
            return None
        gen_all = DataProto.concat(parts) if len(parts) > 1 else parts[0]
        gen_all.meta_info = dict(gen_batch.meta_info)

        obs, _ = envs.reset_partial(
            kwargs=env_kwargs,
            snapshots=[self._pr_groups[u]['pending'][t]['snap'] for u, t in pending],
        )

        uid_list, traj_list, off_list = [], [], []
        for gi in range(n_fresh_groups):
            u = str(uuid.uuid4())
            members = [str(uuid.uuid4()) for _ in range(n)]
            uid_list += [u] * n
            traj_list += members
            off_list += [0] * n
            self._pr_groups[u] = {'born': self._pr_cycle, 'members': set(members),
                                  'done': {}, 'pending': {}}
        for u, t in pending:
            uid_list.append(u)
            traj_list.append(t)
            off_list.append(int(self._pr_groups[u]['pending'][t]['turn_off']))

        # ---- 3. run the shared turn loop for at most cycle_turns turns ----
        (tbl, ep_rew, ep_len, tool_c, tinfos, rrec) = self._turn_loop(
            gen_batch=gen_all,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            obs=obs,
            uid_batch=np.array(uid_list, dtype=object),
            traj_uid=np.array(traj_list, dtype=object),
            turn_offsets=np.array(off_list, dtype=np.int64),
            max_steps_override=cycle_turns,
        )
        self._dump_rollout_records(rrec)

        # ---- 4. split finished vs unfinished; snapshot the unfinished tail state ----
        B = len(tbl)
        finished = np.zeros(B, dtype=bool)
        for i in range(B):
            last = tbl[i][-1]
            finished[i] = bool(last['env_done']) or bool(last['early_stopped'])
        unfinished_idx = [i for i in range(B) if not finished[i]]
        snaps = envs.snapshot(unfinished_idx) if unfinished_idx else []

        snap_it = iter(snaps)
        for i in range(B):
            u, t = uid_list[i], traj_list[i]
            g = self._pr_groups[u]
            prev = g['pending'].pop(t, None)
            rows = (prev['rows'] + tbl[i]) if prev else tbl[i]
            infos = (prev['infos'] + tinfos[i]) if prev else tinfos[i]
            rew = (prev['ep_rew'] if prev else 0.0) + float(ep_rew[i])
            tools = (prev['tool'] if prev else 0.0) + float(tool_c[i])
            if finished[i]:
                g['done'][t] = {'rows': rows, 'infos': infos, 'ep_rew': rew,
                                'ep_len': float(ep_len[i]), 'tool': tools}
            else:
                g['pending'][t] = {
                    'rows': rows, 'infos': infos, 'snap': next(snap_it),
                    'gen_row': (prev['gen_row'] if prev else gen_all.select_idxs([i])),
                    'turn_off': int(off_list[i]) + len(tbl[i]),
                    'ep_rew': rew, 'ep_len': float(ep_len[i]), 'tool': tools,
                }

        # ---- 5. release complete groups atomically ----
        released = []
        for u in list(self._pr_groups):
            g = self._pr_groups[u]
            if not g['pending'] and len(g['done']) == len(g['members']):
                for t, rec in g['done'].items():
                    released.append((u, t, rec))
                del self._pr_groups[u]

        pend_traj = sum(len(g['pending']) for g in self._pr_groups.values())
        held_done = sum(len(g['done']) for g in self._pr_groups.values())
        ages = [self._pr_cycle - g['born'] for g in self._pr_groups.values() if g['pending']]
        self._partial_metrics = {
            'partial/released_traj': len(released),
            'partial/pending_traj': pend_traj,
            'partial/held_complete_traj': held_done,
            'partial/open_groups': len(self._pr_groups),
            'partial/dropped_stale_groups': len(dropped_groups),
            'partial/fresh_groups': n_fresh_groups,
            'partial/resumed_traj': n_resume,
            'partial/mean_open_group_age': float(np.mean(ages)) if ages else 0.0,
        }
        if not released:
            return None

        total_batch_list = [rec['rows'] for _, _, rec in released]
        total_infos = [rec['infos'] for _, _, rec in released]
        episode_rewards = np.array([rec['ep_rew'] for _, _, rec in released], dtype=np.float32)
        episode_lengths = np.array([rec['ep_len'] for _, _, rec in released], dtype=np.float32)
        tool_callings = np.array([rec['tool'] for _, _, rec in released], dtype=np.float32)
        traj_uid_arr = np.array([t for _, t, _ in released], dtype=object)

        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        out = self._finalize_rollout(total_batch_list, episode_rewards, episode_lengths,
                                     success, traj_uid_arr, tool_callings)
        out.meta_info['partial_metrics'] = self._partial_metrics
        return out

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            async_rollout_manager=None,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            async_rollout_manager: AsyncLLMServerManager (rollout.mode=async only) —
                required by the lean async collector (+env.async_rollout_enable).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Lean async rollout (2026-08-04): trajectory-level concurrency, no turn
        # lockstep. Train uses partial-rollout semantics (same _pr_groups state as the
        # sync PoC); val runs every prompt to completion. Requires rollout.mode=async.
        if bool(self.config.env.get('async_rollout_enable', False)):
            assert async_rollout_manager is not None, \
                "env.async_rollout_enable=true requires rollout.mode=async " \
                "(AsyncLLMServerManager not passed to multi_turn_loop)"
            assert str(self.config.algorithm.adv_estimator) != 'carl', \
                "async rollout does not support CARL (it builds its own rollout tree)"
            assert not self.config.algorithm.filter_groups.enable, \
                "async rollout is incompatible with filter_groups (dynamic sampling)"
            from agent_system.multi_turn_rollout.async_rollout import async_multi_turn_loop
            return async_multi_turn_loop(self, gen_batch, async_rollout_manager,
                                         envs, is_train=is_train)
        # Agentic-RL-CA Phase 2b: CARL builds its own per-prompt rollout tree (n0 phase-1
        # rollouts + n_total-n0 snapshot resumes) — group repetition happens inside the
        # loop, NOT via env.rollout.n (which only sizes the env pool for this arm).
        if is_train and str(self.config.algorithm.adv_estimator) == 'carl':
            assert not self.config.algorithm.filter_groups.enable, \
                "CARL is incompatible with filter_groups (dynamic sampling)"
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.carl_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            return self._finalize_rollout(total_batch_list, total_episode_rewards,
                                          total_episode_lengths, total_success,
                                          total_traj_uid, totoal_tool_callings)

        # Sync partial rollout (PoC): handles its own repeat/packing; incompatible with
        # estimators that group by anchor state across a same-cycle batch (GiGPO) or with
        # dynamic sampling (filter_groups) — both assume all group members share a cycle.
        if is_train and bool(self.config.env.get('partial_rollout_enable', False)):
            # CARL builds its own per-prompt rollout tree (snapshot/resume from inside the
            # loop) — incompatible. GiGPO IS compatible: its grouping unit is the uid group
            # (episode advantage over the group, step advantage over anchor-matched turns
            # WITHIN a uid group), and partial rollout releases uid groups ATOMICALLY, so
            # every released batch contains each group whole, with all of its turns.
            # (Measured 2026-08-03: 51.9% of non-empty-anchor turns share an anchor with a
            # sibling, so the step-level term is non-degenerate in this env.)
            assert str(self.config.algorithm.adv_estimator) != 'carl', \
                "partial rollout does not support CARL (it builds its own rollout tree)"
            assert not self.config.algorithm.filter_groups.enable, \
                "partial rollout PoC is incompatible with filter_groups (dynamic sampling)"
            return self.partial_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        return self._finalize_rollout(total_batch_list, total_episode_rewards,
                                      total_episode_lengths, total_success,
                                      total_traj_uid, totoal_tool_callings)

    def _finalize_rollout(self, total_batch_list, total_episode_rewards,
                          total_episode_lengths, total_success, total_traj_uid,
                          totoal_tool_callings) -> DataProto:
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)


        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )

        # SP4 HISR STAGE-4 (guarded; OFF by default → zero effect on other conditions). When
        # HISR_FUSED_REWARD=1, rewrite the per-turn `rewards` with HISR fused process rewards
        # (SPRM × hindsight importance, fused with grounding) before gae_turn consumes them.
        import os as _os, sys as _sys
        if _os.environ.get("HISR_FUSED_REWARD") == "1":
            try:
                # Ray actors may not inherit the shell PYTHONPATH; ensure reward_models is importable.
                _rm = _os.environ.get("HISR_REWARD_MODELS_PATH", "/home/tiger/xiaoxuan/reward_models")
                if _rm not in _sys.path:
                    _sys.path.insert(0, _rm)
                from src.hisr.runtime import apply_hisr_to_rollout_batch
                gen_batch_output = apply_hisr_to_rollout_batch(gen_batch_output, self.tokenizer)
            except Exception as _e:
                print(f"[HISR] reward rewrite FAILED ({_e}); falling back to raw rewards", flush=True)

        return gen_batch_output
