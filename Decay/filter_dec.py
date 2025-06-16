import re
import json
from functools import reduce
from copy import deepcopy
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import Dict, List
from pathlib import Path
from numpy import float64
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

            flat_state.sort()

            if flat_state not in final_states:
                final_states.append(flat_state)
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
with open("final_states_B_0s.json", "w", encoding="utf-8") as f:
    json.dump(find_final_states("B_s0",decays1), f, indent=4, ensure_ascii=False)
with open("final_states_anty_B_0s.json", "w", encoding="utf-8") as f:
    json.dump(find_final_states("anti-B_s0",decays1), f, indent=4, ensure_ascii=False)
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
final_state_particle = list(set(reduce(lambda x,y: x+y, find_final_states('B_s0', decays1) + find_final_states('anti-B_s0', decays1))))
with open("final_state_particle.json", "w", encoding="utf-8") as f:
    json.dump(final_state_particle, f, indent=4, ensure_ascii=False)
# Этап 2 "Создаем все возможные пути"
with open("decays2_anti-B_s0.json", "r", encoding="utf-8") as f:
    decays22 = json.load(f)

with open("decays2_B_s0.json", "r", encoding="utf-8") as f:
    decays21 = json.load(f)
def gen_part_to_int(*decays: Dict[str, List[Dict]]) -> Dict[str, int]:
    part_to_int = {}
    i = 0
    for decay in decays:
        for particle in decay:
            if particle not in part_to_int:
                part_to_int[particle] = i
                i += 1
    return part_to_int
with open("part_to_int.json", "w", encoding="utf-8") as f:
    json.dump(gen_part_to_int(decays21, decays22), f, indent=4, ensure_ascii=False)

part_to_int = gen_part_to_int(decays21, decays22)
def final_path_to_file_indexed(particle: str, decay_dict: Dict, file_handle, particle_to_idx: Dict[str, int], root: bool = True) -> List[Dict]:
    if particle not in decay_dict:
        return [{}]
    if particle in final_state_particle:
        return [{}]

    paths = []
    pid = particle_to_idx[particle]

    for i, decay in enumerate(decay_dict[particle]):
        sub_decay_paths = [
            final_path_to_file_indexed(p, decay_dict, file_handle, particle_to_idx, root=False)
            for p in decay["products"]
        ]
        for combo in product(*sub_decay_paths):
            combined = {pid: i}
            for sub_dict in combo:
                combined.update(sub_dict)
            if root:
                file_handle.write(json.dumps(combined) + "\n")
                file_handle.flush()
            paths.append(combined)

    return paths

def flatten_with_trace(products, depth=0):
    global decay_dict, decay_dict_sup , max_depth
    if depth > max_depth:
        print(f"[!] Max depth {max_depth} reached at products: {products}")
        return [([], [])]
    paths = []
    for p in products:
        if p in final_state_particle:
            paths.append([([p], [])])
        elif (p in decay_dict_sup) :
            local_paths = []
            for decay in decay_dict_sup[p]:
                sub_paths = flatten_with_trace(
                    decay["products"],
                    depth = depth + 1
                )
                for sub_decay, sub_used in sub_paths:
                    local_paths.append((
                        sub_decay,
                        sub_used + [(p, decay["products"])]
                    ))
            paths.append(local_paths)
        elif (p in decay_dict) :
            local_paths = []
            for decay in decay_dict[p]:
                sub_paths = flatten_with_trace(
                    decay["products"],
                    depth = depth + 1
                )
                for sub_decay, sub_used in sub_paths:
                    local_paths.append((
                        sub_decay,
                        sub_used + [(p, decay["products"])]
                    ))
            paths.append(local_paths)
        else:
            paths.append([([p], [])])
    results = []
    for combo in product(*paths):
        flat_decay = sum([x[0] for x in combo], [])
        used = sum([x[1] for x in combo], [])
        results.append((flat_decay, used))
    return results

def match_decay_with_trace(final_states, candidates_with_trace):
    final_states = [Counter(fs) for fs in final_states]
    matched_traces = []
    for decay, used in candidates_with_trace:
        dc = Counter(decay)
        if any(dc == fs for fs in final_states):
            matched_traces.append(used)
    return matched_traces

def process_decay(decay, particle, final_states):
    flat_paths = flatten_with_trace(decay["products"])
    matched = match_decay_with_trace(final_states, flat_paths)
    if matched:
        local_result = defaultdict(list)
        local_result[particle].append(decay)
        for trace in matched:
            for p, prods in trace:
                key = {'products': prods, 'model': 'PHOTOS', 'branching_ratio': 0.0}
                if key not in local_result[p]:
                    local_result[p].append(key)
        return local_result
    return None

def merge_results(results):
    final = defaultdict(list)
    for r in results:
        if not r:
            continue
        for k, vlist in r.items():
            for v in vlist:
                if v not in final[k]:
                    final[k].append(v)
    return dict(final)

def filter_decays(particle, final_states):
    from concurrent.futures import ThreadPoolExecutor
    global decay_dict
    def task(decay):
        return process_decay(decay, particle, final_states)
    decays = decay_dict.get(particle, [])
    results = []
    with ThreadPoolExecutor(max_workers = 50) as executor:
        futures = [executor.submit(task, decay) for decay in decays]
        for future in as_completed(futures):
            results.append(future.result())
    return merge_results(results)

path = Path("/gpfs/home/belle2/matrk/Extend/Decays/")

max_depth=10
decays11.pop('B_s0')
decays12.pop('anti-B_s0')
decay_dict = decays21
decay_dict_sup = decays11
filtered1 = filter_decays("B_s0", find_final_states("B_s0", decays1))
with open(path/"filtered_decays1.json", "w", encoding="utf-8") as f:
    json.dump(filtered1, f, indent=4, ensure_ascii=False)
filtered1
decay_dict = decays22
decay_dict_sup = decays12
filtered2 = filter_decays("anti-B_s0", find_final_states("anti-B_s0", decays1))
with open(path/"filtered_decays2.json", "w", encoding="utf-8") as f:
    json.dump(filtered2, f, indent=4, ensure_ascii=False)
filtered2
with open(path/'decays2_gen_path_anti-B_s0.json', 'w') as f:
     final_path_to_file_indexed('anti-B_s0', decays22, f, part_to_int)
with open(path/'decays2_gen_path_B_s0.json', 'w') as f:
     final_path_to_file_indexed('B_s0', decays21, f, part_to_int)
def merge_decay_dicts(*dicts):
    """Объединяет несколько decay-словрей, избегая дубликатов"""
    merged = defaultdict(list)
    
    for decay_dict in dicts:
        for particle, decays in decay_dict.items():
            for decay in decays:
                if decay not in merged[particle]:
                    merged[particle].append(decay)
                    
    return dict(merged)
new_dec = merge_decay_dicts(filtered1, filtered2)

def update_branching_ratios(target_dict, source_dict):
    for particle, decays in target_dict.items():
        if particle not in source_dict:
            continue
        for decay in decays:
            for src_decay in source_dict[particle]:
                if (sorted(decay["products"]) == sorted(src_decay["products"])):
                    decay["branching_ratio"] = src_decay["branching_ratio"]
                    decay["model"] = src_decay["model"]
                    break  # переходим к следующему decay
    return target_dict

new_dec = update_branching_ratios(new_dec, decays2)
def decay_dict_to_evtgen_format(decay_dict):
    lines = []
    for particle, decays in decay_dict.items():
        lines.append(f"Decay {particle}")
        norm = 0
        for d in decays:
            #print(d["branching_ratio"])
            norm += float64(d["branching_ratio"])
        #print(particle, norm)

        for d in decays:
            products = " ".join(d["products"])
            model = d["model"]
            br = str(float64(d["branching_ratio"])/norm)
            lines.append(f"{br}\t{products}\t{model}")
        lines.append("Enddecay\n")
    return "\n".join(lines)
with open(path/"decays_evtgen_format.txt", "w", encoding="utf-8") as f:
    f.write(decay_dict_to_evtgen_format(new_dec))