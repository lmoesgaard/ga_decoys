from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import subprocess
import pandas as pd
import io

descriptor_list = [
    {"Description": "HeavyAtomCount", "Function": Descriptors.HeavyAtomCount, "dtype": int},
    {"Description": "MolWt", "Function": Descriptors.MolWt, "dtype": float},
    {"Description": "LogP", "Function": Descriptors.MolLogP, "dtype": float},
    {"Description": "NumHAcceptors", "Function": Descriptors.NumHAcceptors, "dtype": int},
    {"Description": "NumHDonors", "Function": Descriptors.NumHDonors, "dtype": int},
    {"Description": "TPSA", "Function": Descriptors.TPSA, "dtype": float},
    {"Description": "NumValenceElectrons", "Function": Descriptors.NumValenceElectrons, "dtype": int},
    {"Description": "NumRotatableBonds", "Function": Descriptors.NumRotatableBonds, "dtype": int},
]


def get_prop_arr(m: Chem.Mol) -> np.ndarray:
    """Compute descriptor array for a molecule in the order of descriptor_list."""
    return np.array([spec["Function"](m) for spec in descriptor_list], dtype=float)

def get_actual_formal_charge(smiles):
    """
    Get actual formal charge from SMILES.
    Returns (positive_count, negative_count) from explicit charges like [NH3+], [O-].
    Does NOT predict pH 7 ionization - just reads what's in the SMILES/Mol.
    Ignores directly bonded (+/-) pairs (e.g. nitro).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (0, 0)

        # collect charged atoms
        pos_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() > 0]
        neg_atoms = [a for a in mol.GetAtoms() if a.GetFormalCharge() < 0]

        # start with raw counts
        n_positive = sum(a.GetFormalCharge() for a in pos_atoms)
        n_negative = -sum(a.GetFormalCharge() for a in neg_atoms)

        # now remove "internal" charge pairs that are directly bonded
        for pa in pos_atoms:
            for nb in pa.GetNeighbors():
                if nb.GetFormalCharge() < 0:
                    # subtract one from both counts
                    n_positive -= 1
                    n_negative -= 1

        return (max(n_positive, 0), max(n_negative, 0))
    except Exception:
        return (None, None)
    
def tautomerize(
        smiles, pH, 
        cutoff, tautomer_limit, protomer_limit, 
        cxcalc_exe, molconvert_exe,
        verbose: bool): 

    if verbose: 
        print('Tautomerization')

    # Suppress noisy ChemAxon Java logging unless in verbose mode
    _stderr = None if verbose else subprocess.DEVNULL

    cmd1 =  f'{cxcalc_exe} -g dominanttautomerdistribution -H {pH} -C false -t tautomer-dist'
    output1 = subprocess.check_output(cmd1, shell=True, input=smiles.encode(), stderr=_stderr)

    cmd2 = f'{molconvert_exe} sdf -g -c "tautomer-dist>={tautomer_limit}" '
    output2 = subprocess.check_output(cmd2, shell=True, input=output1, stderr=_stderr) 

    cmd3 = f'{cxcalc_exe} -g microspeciesdistribution -H {pH} -t protomer-dist'
    output3 = subprocess.check_output(cmd3, shell=True, input=output2, stderr=_stderr)

    cmd4 = f'{molconvert_exe} smiles -g -c "protomer-dist>={protomer_limit}" -T name:tautomer-dist:protomer-dist' 
    output4 = subprocess.check_output(cmd4, shell=True, input=output3, stderr=_stderr)
    table = pd.read_csv(io.BytesIO(output4), sep='\t')
    if len(table) == 1: 
        protomers = list(table['#SMILES'])
    else: 
        # remove redundant SMILES, sort by score (highest first)
        table['score'] = table['tautomer-dist']*table['protomer-dist']/100
        prots = {}
        for smi, score in zip(table['#SMILES'], table['score']): 
            if smi in prots: 
                prots[smi] = max(score, prots[smi])
            elif score > cutoff: 
                prots[smi] = score 
        protomers = sorted(prots, key = lambda x: prots[x], reverse=True)

    return protomers[0]