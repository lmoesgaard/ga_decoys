import random
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from molecule import MoleculeOptions
from molecule import mol_ok, ring_ok


def delete_atom() -> str:
    """ Returns a SMARTS string to delete an atom in a molecule """
    delete_smarts = ['[*:1]~[D1]>>[*:1]',
                     '[*:1]~[C;D2]~[*:2]>>[*:1]-[*:2]']
    p = [0.5, 0.5]

    return np.random.choice(delete_smarts, p=p)


def append_atom() -> str:
    """ Returns a SMARTS string to append an atom to the molecule """
    atom_list = ['C', 'N', 'O', 'F', 'Cl', 'Br']
    p = [1.0 / 6.0] * 6

    new_atom = np.random.choice(atom_list, p=p)

    rxn_smarts = '[#6;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)

    return rxn_smarts


def insert_atom() -> str:
    """ Returns a SMARTS string to insert an atom in a molecule """
    atom_list = ['C', 'N', 'O']
    p = [1.0 / 3.0] * 3
    new_atom = np.random.choice(atom_list, p=p)

    rxn_smarts = '[C:1]!:[C:2]>>[*:1]-X-[*:2]'.replace('X', new_atom)

    return rxn_smarts


def change_bond_order() -> str:
    """ Returns a SMARTS string to change a bond order """
    choices = ['[C:1]!-[C:2]>>[*:1]-[*:2]', '[C;!H0:1]-[C;!H0:2]>>[*:1]=[*:2]',
               '[C:1]#[C:2]>>[*:1]=[*:2]', '[C;!R;!H1;!H0:1]~[C:2]>>[*:1]#[*:2]']
    probabilities = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=probabilities)


def change_atom(mol: Chem.Mol) -> str:
    """ Returns a SMARTS string to change an atom in a molecule to a different one """
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    probabilities = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    valid_choices = []
    for atom_type in choices:
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[' + atom_type + ';D1]-[#6]')):
            valid_choices.append(atom_type)

    if not valid_choices:
        return '[#0]>>[#0]'

    valid_probs = []
    for choice in valid_choices:
        idx = choices.index(choice)
        valid_probs.append(probabilities[idx])

    valid_probs = np.array(valid_probs)
    valid_probs /= valid_probs.sum()

    first = np.random.choice(valid_choices, p=valid_probs)

    second = np.random.choice(choices, p=probabilities)
    while second == first:
        second = np.random.choice(choices, p=probabilities)

    return '[X;D1;$(*-[#6]):1]>>[Y:1]'.replace('X', first).replace('Y', second)


def mutate(mol: Chem.Mol, mutation_rate: float, molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
    """ Mutates a molecule based on actions

    :param mol: the molecule to mutate
    :param mutation_rate: the mutation rate
    :param molecule_options: any filters that should be applied
    :returns: A valid mutated molecule or none if it was not possible
    """
    if random.random() > mutation_rate:
        return mol

    Chem.Kekulize(mol, clearAromaticFlags=True)
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    for i in range(10):
        rxn_smarts_list = 5 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_atom()
        rxn_smarts_list[3] = change_atom(mol)
        rxn_smarts_list[4] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=probabilities)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_molecules: List[Chem.Mol] = []
        for m in new_mol_trial:
            m = m[0]
            if mol_ok(m, molecule_options) and ring_ok(m):
                new_molecules.append(m)

        if len(new_molecules) > 0:
            return np.random.choice(new_molecules)

    return None
