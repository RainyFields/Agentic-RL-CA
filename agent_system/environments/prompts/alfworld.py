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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# SP3.1: "weak admissible-command" prompt (no <think>/<action>; output only the action text).
ALFWORLD_TEMPLATE_WEAK_NO_HIS = """You are an agent operating in the ALFRED Embodied Environment.

Observation:
{current_observation}

Admissible actions:
{admissible_actions}

Choose exactly one admissible action. Output only the action text, nothing else."""

ALFWORLD_TEMPLATE_WEAK = """You are an agent operating in the ALFRED Embodied Environment.

Task:
{task_description}

Recent history (last {history_length} steps):
{action_history}

Observation (step {current_step}):
{current_observation}

Admissible actions:
{admissible_actions}

Choose exactly one admissible action. Output only the action text, nothing else."""

# SP6 ReAct track: matches the expert BC data (alfworld_sft.json) instruction so pi_base is in-distribution
# at rollout. ReAct = "Thought: ... \nAction: ...". Grounding feedback "Nothing happened" = invalid action.
_REACT_PREAMBLE = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".

The available actions are:
1. go to {{recep}}
2. take {{obj}} from {{recep}}
3. put {{obj}} in/on {{recep}}
4. open {{recep}}
5. close {{recep}}
6. toggle {{obj}} {{recep}}
7. clean {{obj}} with {{recep}}
8. heat {{obj}} with {{recep}}
9. cool {{obj}} with {{recep}}
where {{obj}} and {{recep}} correspond to objects and receptacles. Use exact action formats and never simplify; ground strictly in the history; decompose the task accurately.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

ALFWORLD_TEMPLATE_REACT_NO_HIS = _REACT_PREAMBLE + """

{current_observation}"""

ALFWORLD_TEMPLATE_REACT = _REACT_PREAMBLE + """

Your task is to: {task_description}
Below are your most recent {history_length} observations and the actions you took:
{action_history}

Current observation: {current_observation}"""