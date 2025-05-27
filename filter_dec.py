import re
import json
from functools import reduce
from copy import deepcopy
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import Dict, List

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
def merge_decay_dicts(*dicts: Dict) -> Dict:
    merged = {}
    for d in dicts:
        for particle, decays in d.items():
            if particle not in merged:
                merged[particle] = []
            merged[particle].extend(decays)

    # Удалим дубликаты по значению
    for particle in merged:
        merged[particle] = list(set(merged[particle]))

    return merged

def extract_all_products(decay_dict: Dict) -> List[List[str]]:
    return [decay["products"] for decays in decay_dict.values() for decay in decays]


def flatten_with_trace(products, decay_dict, visited=None, depth=0, max_depth=5):
    if visited is None:
        visited = set()
    if depth > max_depth:
        print(f"[!] Max depth {max_depth} reached at products: {products}")
        return [([], [])]

    paths = []
    for p in products:
        if p in visited:
            continue
        if p in decay_dict:
            local_paths = []
            for decay in decay_dict[p]:
                sub_paths = flatten_with_trace(
                    decay["products"],
                    decay_dict,
                    visited | {p},
                    depth + 1,
                    max_depth
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

def process_decay(decay, particle, decay_dict, final_states):
    flat_paths = flatten_with_trace(decay["products"], decay_dict)
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

def filter_decays(particle, decay_dict, final_states):
    results = []
    with ThreadPoolExecutor(max_workers = 50) as executor:
        futures = [
            executor.submit(process_decay, decay, particle, decay_dict, final_states)
            for decay in decay_dict.get(particle, [])
        ]
        for f in as_completed(futures):
            results.append(f.result())
    return merge_results(results)

filtered1 = filter_decays("B_s0", decays21, find_final_states("B_s0", decays1))
filtered2 = filter_decays("anti-B_s0", decays22, find_final_states("anti-B_s0", decays1))

with open("filtered_decays1.json", "w", encoding="utf-8") as f:
    json.dump(filtered1, f, indent=4, ensure_ascii=False)

with open("filtered_decays2.json", "w", encoding="utf-8") as f:
    json.dump(filtered2, f, indent=4, ensure_ascii=False)
