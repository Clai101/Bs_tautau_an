import re
import json
from functools import reduce
from copy import deepcopy
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import Dict, List



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
    for decay in decay_dict.get(particle, []):
        result = process_decay(decay, particle, decay_dict, final_states)
        results.append(result)
    return merge_results(results)


filtered1 = filter_decays("B_s0", decays21, find_final_states("B_s0", decays1))
filtered2 = filter_decays("anti-B_s0", decays22, find_final_states("anti-B_s0", decays1))

with open("filtered_decays1.json", "w", encoding="utf-8") as f:
    json.dump(filtered1, f, indent=4, ensure_ascii=False)

with open("filtered_decays2.json", "w", encoding="utf-8") as f:
    json.dump(filtered2, f, indent=4, ensure_ascii=False)
