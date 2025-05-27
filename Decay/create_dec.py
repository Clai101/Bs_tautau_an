import re
import json
from functools import reduce
from copy import deepcopy
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from typing import Dict, List
from pathlib import Path

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

decays2 = parse_decay_block(text_data2)

path = Path("/gpfs/home/belle2/matrk/Extend/Decays/")
with open(path/"filtered_decays1.json", "r", encoding="utf-8") as f:
    filtered1 = json.loads(f)

with open(path/"filtered_decays2.json", "r", encoding="utf-8") as f:
    filtered2 = json.loads(f)


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

for key, dec in new_dec.items():
    decays2[key] = dec

def decay_dict_to_evtgen_format(decay_dict):
    lines = []
    for particle, decays in decay_dict.items():
        lines.append(f"Decay {particle}")
        for d in decays:
            products = " ".join(d["products"])
            model = d["model"]
            br = d["branching_ratio"]
            lines.append(f"{br:.1f}   {products}   {model}")
        lines.append("Enddecay\n")
    return "\n".join(lines)

dec_text = decay_dict_to_evtgen_format(decays2)

with open(path/"decays_evtgen_format.txt", "w", encoding="utf-8") as f:
    f.write(dec_text)

