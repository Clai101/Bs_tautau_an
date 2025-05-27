import re
import json
from functools import reduce
from copy import deepcopy
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import Dict, List

# Этап 1 "Читаем .DEC"

def make_anti(particle: str) -> str:
    #print(particle)
    wronf_p = {"anti-Lambda_c-": "Lambda_c+", "Lambda_c(2593)+":"anti-Lambda_c(2593)-", "Lambda_c(2625)+":"anti-Lambda_c(2625)-"}
    if particle in wronf_p:
        return wronf_p[particle]
    if particle in ["pi0", "rho0", "K_S0"]:
        return particle
    if particle.endswith("+"):
        return particle[:-1] + "-"
    if particle.endswith("-"):
        return particle[:-1] + "+"
    if "0" in particle:
        if particle.startswith("anti-"):
            return particle[5:]
        return "anti-" + particle
    return particle

def parse_decay_block(text: str) -> Dict[str, List[Dict]]:
    """
    Парсит текст в формате DECAY BLOCK в словарь с продуктами, BR и моделью.
    Также обрабатывает директиву CDecay для автогенерации сопряжённых распадов.
    """
    decay_dict = {}
    current_decay = None
    cdecay_links = []

    MODEL_KEYWORDS = {
        'PHOTOS', 'ISGW2', 'PHSP', 'SVS', 'STS', 'PYTHIA',
        'TAULNUNU', 'TAUSCALARNU', 'TAUVECTORNU',
        'SVV_HELAMP', 'VSP_PWAVE'
    }

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith("Decay"):
            parts = line.split()
            if len(parts) > 1:
                current_decay = parts[1]
                decay_dict[current_decay] = []
            continue

        if line.startswith("CDecay"):
            parts = line.split()
            if len(parts) > 1:
                cdecay_links.append(parts[1])
            continue

        if line.startswith("Enddecay"):
            current_decay = None
            continue

        if current_decay:
            line = re.sub(r'#.*', '', line)
            line = line.rstrip(';')
            tokens = line.split()

            if len(tokens) < 2:
                continue

            try:
                br = float(tokens[0])
            except ValueError:
                continue

            model_start = next(
                (i for i, t in enumerate(tokens[1:], 1)
                 if re.fullmatch(r'[A-Z0-9_]+', t) and t in MODEL_KEYWORDS),
                len(tokens)
            )

            products = tokens[1:model_start]
            model = ' '.join(tokens[model_start:])

            decay_dict[current_decay].append({
                'branching_ratio': br,
                'products': products,
                'model': model
            })

    for anti_particle in cdecay_links:

        orig_particle = make_anti(anti_particle)


        anti_decays = []
        for decay in decay_dict[orig_particle]:
            anti_products = [make_anti(p) for p in decay["products"]]
            anti_decays.append({
                'branching_ratio': decay['branching_ratio'],
                'products': anti_products,
                'model': decay['model']
            })
        decay_dict[anti_particle] = anti_decays

    return decay_dict


text_data2 = str()
with open("DECAY_1.DEC", "r") as outfile:
    text_data2 = reduce(lambda x, y: x +y, [i for i in outfile.readlines()])
text_data1 = str()
with open("my_dec.DEC", "r") as outfile:
    text_data1 = reduce(lambda x, y: x +y, [i for i in outfile.readlines()])

decays1 = parse_decay_block(text_data1)
decays2 = parse_decay_block(text_data2)

def find_final_states(particle: str, decay_dict: Dict) ->   List[List[str]]:
    if particle not in decay_dict:
        return [[particle]]

    final_states = []

    for decay in decay_dict[particle]:
        branches = [find_final_states(p, decay_dict) for p in decay["products"]]
        from itertools import product
        for combination in product(*branches):
            flat_state = []
            for group in combination:
                flat_state.extend(group)
            final_states.append(flat_state)
    for final in final_states:
        final = final.sort()
    return final_states

def prune_unreachable_particles(decay_dict: Dict, root: str) -> Dict:
    reachable = set()
    frontier = {root}

    while frontier:
        next_frontier = set()
        for particle in frontier:
            if particle not in decay_dict:
                continue
            reachable.add(particle)
            for decay in decay_dict[particle]:
                for p in decay["products"]:
                    if p not in reachable:
                        next_frontier.add(p)
        frontier = next_frontier

    pruned_dict = {p: d for p, d in decay_dict.items() if p in reachable}
    return pruned_dict


decays21 = prune_unreachable_particles(decays2, "B_s0")
decays22 = prune_unreachable_particles(decays2, "anti-B_s0")
decays11 = prune_unreachable_particles(decays1, "B_s0")
decays12 = prune_unreachable_particles(decays1, "anti-B_s0")

with open("decays1_B_s0.json", "w", encoding="utf-8") as f:
    json.dump(decays11, f, indent=4, ensure_ascii=False)

with open("decays1_anti-B_s0.json", "w", encoding="utf-8") as f:
    json.dump(decays12, f, indent=4, ensure_ascii=False)

with open("decays2_B_s0.json", "w", encoding="utf-8") as f:
    json.dump(decays21, f, indent=4, ensure_ascii=False)

with open("decays2_anti-B_s0.json", "w", encoding="utf-8") as f:
    json.dump(decays22, f, indent=4, ensure_ascii=False)

# Этап 2 "Создаем все возможные пути"
with open("decays2_anti-B_s0.json", "r", encoding="utf-8") as f:
    decays22 = json.load(f)

with open("decays1_B_s0.json", "r", encoding="utf-8") as f:
    decays21 = json.load(f)
def final_path_to_file(particle: str, decay_dict: Dict, file_handle, root: bool = True) -> List[Dict]:
    if particle not in decay_dict:
        return [{}]

    paths = []
    for i, decay in enumerate(decay_dict[particle]):
        sub_decay_paths = [final_path_to_file(p, decay_dict, file_handle, root=False) for p in decay["products"]]
        for combo in product(*sub_decay_paths):
            combined = {particle: i}
            for sub_dict in combo:
                combined.update(sub_dict)
            if root:
                file_handle.write(json.dumps(combined) + "\n")
                file_handle.flush()
            paths.append(combined)

    return paths
with open('decays2_gen_path_anti-B_s0.json', 'w') as f:
     final_path_to_file('anti-B_s0', decays22, f)
with open('decays2_gen_path_B_s0.json', 'w') as f:
     final_path_to_file('B_s0', decays21, f)