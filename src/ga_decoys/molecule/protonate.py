import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="to-Python converter for boost::shared_ptr",
)

import subprocess
import pandas as pd
import io
import json
import sys
import time
import os
from rdkit import Chem


def sdf_to_df(f):
    df = []
    current = {"pose": ""}
    info = False
    for line in [line+"\n" for line in f.split("\n")]:
        if len(current["pose"]) == 0:
            current["name"] = line.strip()
        current["pose"] += line
        if line.startswith(">  <"):
            name = line[4:-2]
            info = True
        elif info:
            try:
                current[name] = float(line.replace(",", ".").strip())
            except:
                current[name] = line.strip()
            info = False
        elif line.startswith("$$$"):
            df.append(current)
            current = {"pose": ""}
    df = pd.DataFrame(df)
    return df

def filter_sdf(file):
    df = sdf_to_df(file)
    if df is None or df.empty:
        return ""
    if "pose" not in df.columns:
        return ""
    if "name" in df.columns:
        df = df.drop_duplicates("name")
    return "".join(df["pose"].astype(str).to_list())

def tautomerize(
        smiles, pH,
        cxcalc_exe,
        verbose: bool):

    if verbose:
        print('Tautomerization')

    # Keep output aligned to the number of input lines.
    input_lines = [l for l in smiles.split("\n") if l.strip()]
    input_names = []
    for l in input_lines:
        parts = l.split()
        input_names.append(parts[1] if len(parts) > 1 else None)

    _stderr = None if verbose else subprocess.DEVNULL

    cmd1 =  f'{cxcalc_exe} -g dominanttautomerdistribution -H {pH} -C false -t tautomer-dist'
    output1 = subprocess.check_output(cmd1, shell=True, input=smiles.encode(), stderr=_stderr)

    output2_text = filter_sdf(output1.decode())
    if not output2_text.strip():
        return [None] * len(input_names)
    output2 = output2_text.encode()

    cmd3 = f'{cxcalc_exe} -g microspeciesdistribution -H {pH} -t protomer-dist'
    output3 = subprocess.check_output(cmd3, shell=True, input=output2, stderr=_stderr)

    output4 = filter_sdf(output3.decode())
    if not output4.strip():
        return [None] * len(input_names)

    table = sdf_to_df(output4)
    if table is None or table.empty or "pose" not in table.columns or "name" not in table.columns:
        return [None] * len(input_names)

    def _molblock_to_smiles(molblock: str):
        try:
            mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=True)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    table['#SMILES'] = table.pose.apply(_molblock_to_smiles)
    # Keep only rows that successfully converted.
    table = table[table['#SMILES'].notnull()]
    if table.empty:
        return [None] * len(input_names)

    for col in ['tautomer-dist', 'protomer-dist']:
        if col not in table.columns:
            table[col] = None

    table = table[['#SMILES', 'name', 'tautomer-dist', 'protomer-dist']]
    protomers = table.set_index("name").to_dict().get('#SMILES', {})
    return [protomers.get(name, None) for name in input_names]


def protonate_smiles(smi_lst, config=None):
    if config is None:
        config = os.path.join(os.path.dirname(__file__), "default_protonate.json")
    with open(config, 'r') as f:
        ca_parms = json.load(f)

    # cxcalc expects "SMILES name" per line.
    smiles = "\n".join(f"{smi} lig{i}" for i, smi in enumerate(smi_lst)) + "\n"
    protomers = tautomerize(
        smiles, pH=ca_parms["pH"], cxcalc_exe=ca_parms["cxcalc_exe"],
        verbose=ca_parms.get("verbose", False))
    return protomers


if __name__ == "__main__":
    assert len(sys.argv) >= 2, "Usage: protonate.py input.smiles [output.smi] [config.json]"
    
    smiles = sys.argv[1]
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    else:
        outfile = 'output.smi'

    if len(sys.argv) > 3:
        config_file = sys.argv[3]
    else:
        config_file = os.path.join(os.path.dirname(__file__), "default_protonate.json")

    with open(config_file, 'r') as f:
        ca_parms = json.load(f)

    with open(smiles, 'r') as f:
        smiles = f.read()

    t0 = time.time()
    protomers = tautomerize(
        smiles, pH=ca_parms["pH"], cxcalc_exe=ca_parms["cxcalc_exe"],
        verbose=ca_parms.get("verbose", False))

    with open(outfile, "w") as f:
        for protomer in protomers:
            f.write(f"{protomer}\n")