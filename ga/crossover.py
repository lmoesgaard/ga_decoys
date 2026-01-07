import random
from typing import List, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from molecule import mol_ok
from molecule import MoleculeOptions


def cut(mol: Chem.Mol) -> Union[None, List[Chem.Mol]]:
    """ Cuts a single bond that is not in a ring """
    smarts_pattern = "[C]-;!@[*]"
    if not mol.HasSubstructMatch(Chem.MolFromSmarts(smarts_pattern)):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))  # single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        fragments: List[Chem.Mol] = Chem.GetMolFrags(fragments_mol, asMols=True)
    except ValueError:  # CSS: I have no idea what exception can be thrown here
        return None
    else:
        return fragments


def crossover_non_ring(parent_a: Chem.Mol,
                       parent_b: Chem.Mol,
                       molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
    for i in range(10):
        fragments_a = cut(parent_a)
        fragments_b = cut(parent_b)
        if fragments_a is None or fragments_b is None:
            return None

        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_a:
            for fb in fragments_b:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_molecules = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol, molecule_options):
                new_molecules.append(mol)

        if len(new_molecules) > 0:
            return random.choice(new_molecules)

    return None


def crossover(parent_a: Chem.Mol,
              parent_b: Chem.Mol,
              molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
    if parent_a is None or parent_b is None:
        return None
    parent_smiles = [Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)]
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except ValueError:  # CSS: I have no idea about what errors can be thrown here?
        pass
    for i in range(10):
        new_mol = crossover_non_ring(parent_a, parent_b, molecule_options)
        if new_mol is not None:
            new_smiles = Chem.MolToSmiles(new_mol)
            if new_smiles not in parent_smiles:
                return new_mol

    return None
