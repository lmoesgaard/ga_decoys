import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="to-Python converter for boost::shared_ptr")
from multiprocessing import Pool
import random
from typing import List, Tuple, Dict
import time
import json
import argparse
import os
import hashlib
import re

from rdkit import Chem
import numpy as np
from rdkit import rdBase
from molecule.descriptors import descriptor_list, get_actual_formal_charge, get_prop_arr
from filtering import Filtering
from molecule import protonate_smiles

import ga
import molecule

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_scoring_options(smiles: str, config: dict, sample_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    prop_array = get_prop_arr(mol)
    w = [config.get(descriptor["Description"], 0) / std for std, descriptor in zip(sample_std, descriptor_list)]
    return np.array(prop_array), np.array(w)


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)


def read_smiles_lines(path: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"lig{idx}"
            entries.append((smiles, name))
    return entries


def safe_filename(name: str) -> str:
    # Keep filenames portable: replace problematic characters with underscores.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "lig"


def deterministic_seed(smiles: str) -> int:
    # Stable across runs and processes (unlike Python's built-in hash()).
    digest = hashlib.md5(smiles.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little', signed=False)


def score(input_population: List[Chem.Mol], scoring_options: Tuple[np.ndarray, np.ndarray], molecule_options: molecule.MoleculeOptions) -> Tuple[List[Chem.Mol], List[float]]:
    target_prop_array, w = scoring_options
    scores = []
    for mol in input_population:
        if mol is None:
            scores.append(0.0)
            continue
        prop_array = get_prop_arr(mol)
        diff = prop_array - target_prop_array
        error = np.sum(np.abs(diff)*w)
        fitness = 1.0 / (1.0 + error)
        scores.append(fitness)

    # Protonate only valid molecules but keep list alignment.
    indices = []
    smiles_to_protonate = []
    for i, mol in enumerate(input_population):
        if mol is None:
            continue
        try:
            smiles_to_protonate.append(Chem.MolToSmiles(mol))
            indices.append(i)
        except Exception:
            continue

    protonated_smiles_full = [None] * len(input_population)
    if smiles_to_protonate:
        protonated_smiles = protonate_smiles(smiles_to_protonate)
        for i, smi in zip(indices, protonated_smiles):
            protonated_smiles_full[i] = smi

    charges = [get_actual_formal_charge(s) if s is not None else (None, None) for s in protonated_smiles_full]
    scores = [
        score * int(molecule_options.desired_charge[0] == charge[0] and molecule_options.desired_charge[1] == charge[1])
        for score, charge in zip(scores, charges)
    ]
    input_population = [Chem.MolFromSmiles(smi) if smi is not None else None for smi in protonated_smiles_full]
    return input_population[:], scores


def print_list(value: List[float], name: str) -> None:
    s = f"{name:s}:"
    for v in value:
        s += f"{v:6.2f} "
    print(s)


def gbga(ga_opt: ga.GAOptions, mo_opt: molecule.MoleculeOptions, scoring_options: Dict[str, Tuple[float, float]]) -> Tuple[List[Chem.Mol], List[float]]:
    try:
        np.random.seed(ga_opt.random_seed)
        random.seed(ga_opt.random_seed)

        initial_population = ga.make_initial_population(ga_opt)
        population, scores = score(initial_population, scoring_options, mo_opt)

        if len(population) == 0:
            return [], []

        for generation in range(ga_opt.num_generations):
            if len(population) == 0:
                break

            mating_pool = ga.make_mating_pool(population, scores, ga_opt)
            if len(mating_pool) == 0:
                break
            initial_population = ga.reproduce(mating_pool, ga_opt, mo_opt)

            new_population, new_scores = score(initial_population, scoring_options, mo_opt)
            population, scores = ga.sanitize(population+new_population, scores+new_scores, ga_opt)

            if len(population) == 0:
                break

        # Apply final stricter tanimoto cutoff if specified
        if ga_opt.final_tanimoto_cutoff is not None and ga_opt.final_tanimoto_cutoff != ga_opt.tanimoto_cutoff:
            final_opt = ga.GAOptions(
                ga_opt.input_filename, ga_opt.basename, ga_opt.num_generations,
                ga_opt.population_size, ga_opt.mating_pool_size, ga_opt.mutation_rate,
                ga_opt.max_score, ga_opt.random_seed, ga_opt.prune_population,
                tanimoto_cutoff=ga_opt.final_tanimoto_cutoff,
                final_tanimoto_cutoff=ga_opt.final_tanimoto_cutoff,
                target_smiles=ga_opt.target_smiles
            )
            population, scores = ga.sanitize(population, scores, final_opt)

        return population, scores
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [], []


def gbga_for_smiles(input_smiles: str,
                    input_name: str,
                    ga_opt: ga.GAOptions,
                    mo_opt: molecule.MoleculeOptions,
                    scoring_options: Tuple[np.ndarray, np.ndarray]):
    pop, scores = gbga(ga_opt, mo_opt, scoring_options)
    return input_smiles, input_name, pop, scores


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm for Decoy Generation')
    parser.add_argument('--input', type=str, required=True, help='Input .smi file (one SMILES per line)')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'config/test.json'), help='Configuration JSON file')
    args = parser.parse_args()

    config = load_config(args.config)

    input_entries = read_smiles_lines(args.input)
    if len(input_entries) == 0:
        raise SystemExit(f"No SMILES found in {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    population_size = config.get('population_size', 100) 
    mating_pool_size = config.get('mating_pool_size', 100)
    generations = config.get('generations', 10)
    mutation_rate = config.get('mutation_rate', 0.25)
    n_cpus = config.get('n_cpus', 6)
    basename = config.get('basename', args.outdir)
    max_molecule_size = config.get('max_molecule_size', 40)
    tanimoto_cutoff = config.get('tanimoto_cutoff', None)
    # Use relaxed cutoff during optimization, apply user's cutoff at the end
    optimization_cutoff = 0.65 if tanimoto_cutoff is not None else None

    sampled_std_path = config.get("sampled_std", os.path.join(SCRIPT_DIR, "data/sampled_std.npy"))
    if not os.path.isabs(sampled_std_path):
        sampled_std_path = os.path.join(SCRIPT_DIR, sampled_std_path)
    sample_std = np.load(sampled_std_path)

    print('* RDKit version', rdBase.rdkitVersion)
    print('* population_size', population_size)
    print('* mating_pool_size', mating_pool_size)
    print('* generations', generations)
    print('* mutation_rate', mutation_rate)
    print('* max_molecule_size', max_molecule_size)
    print('* input', args.input)
    print('* outdir', args.outdir)
    print('* number of inputs', len(input_entries))
    print('* number of CPUs', n_cpus)
    print('* ')
    print('input_index,input_name,input_smiles,best_score,best_smiles,generations,representation,prune')

    t0 = time.time()
    
    pool_args = []
    for smiles, name in input_entries:
        charge = get_actual_formal_charge(smiles)
        molecule_filters = {"Charge": charge, "PAINS": True, "BRENK": True}
        file_name = os.path.join(SCRIPT_DIR, "input_smiles/{}_{}.smi".format(*charge))
        scoring_options = get_scoring_options(smiles, config, sample_std)

        ga_opt = ga.GAOptions(
            file_name,
            basename,
            generations,
            population_size,
            mating_pool_size,
            mutation_rate,
            9999.0,
            deterministic_seed(smiles),
            True,
            tanimoto_cutoff=optimization_cutoff,
            final_tanimoto_cutoff=tanimoto_cutoff,
            target_smiles=smiles,
        )
        mo_opt = molecule.MoleculeOptions(
            max_molecule_size=max_molecule_size,
            molecule_filters=molecule_filters,
            substructure_filter=Filtering(molecule_filters),
            desired_charge=charge,
        )
        pool_args.append((smiles, name, ga_opt, mo_opt, scoring_options))

    with Pool(n_cpus) as pool:
        output: List = pool.starmap(gbga_for_smiles, pool_args)

    results_path = os.path.join(args.outdir, 'results.csv')
    with open(results_path, 'w') as out_f:
        out_f.write('input_index,input_name,input_smiles,best_score,best_smiles,generations,representation,prune\n')
        for idx, (input_smiles, input_name, pop, scores) in enumerate(output):
            base = safe_filename(input_name)

            # Per-input outputs
            csv_path = os.path.join(args.outdir, f"{base}.csv")
            smi_path = os.path.join(args.outdir, f"{base}.smi")

            with open(csv_path, 'w') as csv_f:
                csv_f.write('score,smiles\n')
                for mol, sc in sorted(zip(pop, scores), key=lambda t: t[1], reverse=True):
                    if mol is None:
                        continue
                    csv_f.write(f"{sc:.6f},{Chem.MolToSmiles(mol)}\n")

            with open(smi_path, 'w') as smi_f:
                for j, mol in enumerate(pop):
                    if mol is None:
                        continue
                    smi = Chem.MolToSmiles(mol)
                    smi_f.write(f"{smi} {input_name}_{j}\n")

            # Global summary row
            if pop:
                best_idx = scores.index(max(scores))
                best_smiles = Chem.MolToSmiles(pop[best_idx])
                best_score = max(scores)
                line = f"{idx:d},{input_name:s},{input_smiles:s},{best_score:.6f},{best_smiles:s},{generations:d},GBGA,True\n"
            else:
                line = f"{idx:d},{input_name:s},{input_smiles:s},,No valid molecules found,{generations:d},GBGA,True\n"
            print(line.strip())
            out_f.write(line)

    t1 = time.time()
    print("")
    print("time = {0:.2f} minutes".format((t1-t0)/60.0))


if __name__ == '__main__':
    main()