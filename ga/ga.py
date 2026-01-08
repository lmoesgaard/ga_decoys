from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molecule import MoleculeOptions
from .crossover import crossover
from .mutation import mutate
from .util import calculate_normalized_fitness, read_smiles_file


@dataclass
class GAOptions:
    input_filename: str
    basename: str
    num_generations: int
    population_size: int
    mating_pool_size: int
    mutation_rate: float
    max_score: float
    random_seed: int
    prune_population: bool
    tanimoto_cutoff: Optional[float] = None
    final_tanimoto_cutoff: Optional[float] = None
    target_smiles: Optional[str] = None


def make_initial_population(options: GAOptions) -> List[Chem.Mol]:
    """ Constructs an initial population from a file with a certain size

        :param options: GA options
        :returns: list of RDKit molecules
    """
    mol_list = read_smiles_file(options.input_filename)
    population: List[Chem.Mol] = []
    for i in range(options.population_size):
        population.append(np.random.choice(mol_list))

    return population


def make_mating_pool(population: List[Chem.Mol], scores: List[float], options: GAOptions) -> List[Chem.Mol]:
    """ Constructs a mating pool, i.e. list of molecules selected to generate offspring

        :param population: the population used to construct the mating pool
        :param scores: the fitness of each molecule in the population
        :param options: GA options
        :returns: list of molecules to use as a starting point for offspring generation
    """
    fitness = calculate_normalized_fitness(scores)
    mating_pool = []
    for i in range(options.mating_pool_size):
        mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool


def reproduce(mating_pool: List[Chem.Mol],
              options: GAOptions,
              molecule_options: MoleculeOptions) -> List[Chem.Mol]:
    """ Creates a new population based on the mating_pool

        :param mating_pool: list of molecules to mate from
        :param options: GA options
        :param molecule_options: Options for molecules
        :returns: a list of molecules that are offspring of the mating_pool
    """
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")
    new_population: List[Chem.Mol] = []
    attempts = 0
    max_attempts = options.population_size * 100  # Safety limit

    while len(new_population) < options.population_size:
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Could not generate full population. Generated {len(new_population)}/{options.population_size} children after {attempts} attempts.")
            break
        
        # Decide whether to do crossover or mutation-only
        if np.random.random() < options.mutation_rate:
            # Mutation-only path: select single parent and mutate (or keep parent if mutation fails)
            parent = np.random.choice(mating_pool)
            new_child = mutate(parent, 1.0, molecule_options)  # Always attempt mutation
            # If mutation fails, accept the parent as-is to keep population growing
            new_population.append(new_child if new_child is not None else parent)
        else:
            # Crossover path: crossover two parents, then maybe mutate
            parent_a = np.random.choice(mating_pool)
            parent_b = np.random.choice(mating_pool)
            new_child = crossover(parent_a, parent_b, molecule_options)
            
            if new_child is not None:
                mutated_child = mutate(new_child, options.mutation_rate, molecule_options)
                # Keep crossover result even if mutation fails
                new_population.append(mutated_child if mutated_child is not None else new_child)
    rdBase.EnableLog("rdApp.error")
    rdBase.EnableLog("rdApp.warning")

    return new_population


def sanitize(population: List[Chem.Mol],
             scores: List[float],
             ga_options: GAOptions) -> Tuple[List[Chem.Mol], List[float]]:
    """ Cleans a population of molecules and returns a sorted list of molecules and scores

    :param population: the list of RDKit molecules to clean
    :param scores: the scores of the molecules
    :param ga_options: GA options
    :return: a tuple of molecules and scores
    """
    if ga_options.prune_population:
        smiles_list = []
        population_tuples = []
        for score, mol in zip(scores, population):
            smiles = Chem.MolToSmiles(mol)
            if smiles not in smiles_list:
                smiles_list.append(smiles)
                population_tuples.append((score, mol))
    else:
        population_tuples = list(zip(scores, population))

    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)

    cutoff = ga_options.tanimoto_cutoff
    if cutoff is not None and ga_options.target_smiles is not None:
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        try:
            target_mol = Chem.MolFromSmiles(ga_options.target_smiles)
            target_fp = None
            if target_mol is not None:
                target_fp = fp_gen.GetFingerprint(target_mol)
        except Exception:
            target_fp = None

        selected: List[Tuple[float, Chem.Mol]] = []
        selected_fps = []

        for score, mol in population_tuples:
            if len(selected) >= ga_options.population_size:
                break
            if mol is None:
                continue

            try:
                fp = fp_gen.GetFingerprint(mol)
            except Exception:
                continue

            max_sim = 0.0
            if target_fp is not None:
                try:
                    max_sim = max(max_sim, DataStructs.TanimotoSimilarity(fp, target_fp))
                except Exception:
                    pass

            for existing_fp in selected_fps:
                sim = DataStructs.TanimotoSimilarity(fp, existing_fp)
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= cutoff:
                    break

            if max_sim < cutoff:
                selected.append((score, mol))
                selected_fps.append(fp)

        # If the cutoff is too strict, fall back to filling remaining slots by score
        # to keep the GA operational.
        if len(selected) < ga_options.population_size:
            selected_smiles = {Chem.MolToSmiles(m) for _, m in selected}
            for score, mol in population_tuples:
                if len(selected) >= ga_options.population_size:
                    break
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol)
                if smi in selected_smiles:
                    continue
                selected.append((score, mol))
                selected_smiles.add(smi)

        population_tuples = selected
    else:
        population_tuples = population_tuples[:ga_options.population_size]

    new_population = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]
    return new_population, new_scores
