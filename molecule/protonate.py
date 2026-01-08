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
    return "".join(df.drop_duplicates("name").pose.to_list())

def tautomerize(
        smiles, pH,
        cxcalc_exe,
        verbose: bool):

    if verbose:
        print('Tautomerization')

    _stderr = None if verbose else subprocess.DEVNULL

    cmd1 =  f'{cxcalc_exe} -g dominanttautomerdistribution -H {pH} -C false -t tautomer-dist'
    output1 = subprocess.check_output(cmd1, shell=True, input=smiles.encode(), stderr=_stderr)

    output2 = filter_sdf(output1.decode()).encode()

    cmd3 = f'{cxcalc_exe} -g microspeciesdistribution -H {pH} -t protomer-dist'
    output3 = subprocess.check_output(cmd3, shell=True, input=output2, stderr=_stderr)

    output4 = filter_sdf(output3.decode())
    table = sdf_to_df(output4)
    table['#SMILES'] = table.pose.apply(lambda x: Chem.MolToSmiles(Chem.MolFromMolBlock(x)))
    table = table[['#SMILES', 'name', 'tautomer-dist', 'protomer-dist']]
    protomers = table.set_index("name").to_dict()['#SMILES']
    return [protomers.get(l.split(" ")[1], None) for l in smiles.split("\n")[:-1]]


def protonate_smiles(smi_lst, config=None):
    if config is None:
        config = os.path.join(os.path.dirname(__file__), "default_protonate.json")
    with open(config, 'r') as f:
        ca_parms = json.load(f)

    smiles = " lig{}\n".join(smi_lst).format(*range(len(smi_lst)))
    protomers = tautomerize(
        smiles, pH=ca_parms["pH"], cxcalc_exe=ca_parms["cxcalc_exe"],
        verbose=ca_parms.get("verbose", False))
    return protomers


if __name__ == "__main__":
    smiles = sys.argv[1]
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
    else:
        config_file = 'default_protonate.json'

    with open(config_file, 'r') as f:
        ca_parms = json.load(f)

    with open(smiles, 'r') as f:
        smiles = f.read()

    t0 = time.time()
    protomers = tautomerize(
        smiles, pH=ca_parms["pH"], cxcalc_exe=ca_parms["cxcalc_exe"],
        verbose=ca_parms.get("verbose", False))

    with open("output.smi", "w") as f:
        for protomer in protomers:
            f.write(f"{protomer}\n")
    print(time.time() - t0)