# SP6 — SHARED transcript formatter for the full-history + one-shot ReAct BC/eval track.
# Imported by BOTH the BC data converter (reward_models) and the env (env_manager) so that a BC
# training prompt is byte-identical in structure to a live rollout prompt. Format (per user spec):
#   system preamble  ->  fixed one-shot example  ->  task  ->  initial obs  ->  full thought/action/obs transcript
import re

PREAMBLE = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles. Use exact action formats and never simplify; ground strictly in the history; decompose the task accurately.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

# fixed one-shot (drawn once, seed=0, src_index=829) — stored as structured data, rendered by the same formatter
_ONESHOT = {'task': 'clean some soapbar and put it in garbagecan.', 'room': 'You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, a towelholder 2, and a towelholder 1.', 'steps': [{'response': 'Thought: To clean the soapbar and dispose of it, I need to locate it first. The task description mentions various items and fixtures in the room, but I need to find where the soapbar is located.\nAction: go to toilet 1', 'next_obs': 'Observation: On the toilet 1, you see a candle 1, a papertowelroll 1, and a soapbar 2.'}, {'response': 'Thought: I have found the soapbar. I need to pick it up to clean it before disposing of it.\nAction: take soapbar 2 from toilet 1', 'next_obs': 'Observation: You pick up the soapbar 2 from the toilet 1.'}, {'response': 'Thought: With the soapbar in hand, I need to find a suitable place to clean it. The sink is typically used for cleaning items, so I should head there.\nAction: go to sinkbasin 2', 'next_obs': 'Observation: On the sinkbasin 2, you see a dishsponge 2.'}, {'response': 'Thought: The sink is equipped for cleaning. I will use the water and possibly the dishsponge if needed to clean the soapbar.\nAction: clean soapbar 2 with sinkbasin 2', 'next_obs': 'Observation: You clean the soapbar 2 using the sinkbasin 2.'}, {'response': 'Thought: The soapbar is now clean. The next step is to dispose of it as instructed. I need to locate the garbage can.\nAction: go to garbagecan 1', 'next_obs': 'Observation: On the garbagecan 1, you see a toiletpaper 2.'}, {'response': 'Thought: I will place the soapbar in the garbage can, completing the task of disposing of it.\nAction: put soapbar 2 in/on garbagecan 1', 'next_obs': None}], 'src_index': 829}


def _strip_welcome(obs):
    return re.sub(r"^\s*-=.*?=-\s*", "", obs.strip(), flags=re.DOTALL).strip()


def fmt_step_obs(obs):
    """A non-initial observation -> single 'Observation: ...' line (normalize prefix)."""
    o = _strip_welcome(obs)
    o = re.sub(r"^Observation:\s*", "", o).strip()
    return "Observation: " + o


def fmt_task_and_room(initial_obs):
    """Initial obs -> 'Your task is to: X' then the room description (task-first, per spec)."""
    o = _strip_welcome(initial_obs)
    m = re.search(r"(.*?)\s*Your task is to:\s*(.+)$", o, re.DOTALL)
    if m:
        room, task = m.group(1).strip(), m.group(2).strip()
        return "Your task is to: %s\n%s" % (task, room)
    return o


def _render_oneshot():
    lines = [ "Your task is to: %s\n%s" % (_ONESHOT["task"], _ONESHOT["room"]) ]
    for st in _ONESHOT["steps"]:
        lines.append(st["response"].strip())
        if st.get("next_obs"):
            lines.append(fmt_step_obs(st["next_obs"]))
    return "\n".join(lines)


ONESHOT_BLOCK = "Here is an example of a completed task:\n\n" + _render_oneshot() + "\n\nNow solve the following task."


def build_transcript_prompt(initial_obs, steps):
    """initial_obs: raw initial observation (room+task). steps: list of (response, next_obs) for each
    COMPLETED turn (last next_obs = current obs). Returns the full user prompt."""
    lines = [ fmt_task_and_room(initial_obs) ]
    for resp, nxt in steps:
        lines.append(resp.strip())
        lines.append(fmt_step_obs(nxt))
    body = "\n".join(lines)
    return "%s\n\n%s\n\n%s" % (PREAMBLE, ONESHOT_BLOCK, body)
