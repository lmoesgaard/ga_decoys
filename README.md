# ga_decoys

Genetic-algorithm decoy generation using RDKit.

Given one or more input ligands (SMILES), this tool evolves *decoy* molecules that try to match the input’s physicochemical descriptor profile while enforcing basic medicinal-chemistry constraints and a diversity cutoff.
## Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/ga_decoys.git
cd ga_decoys

# Install in development mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/yourusername/ga_decoys.git
```

### Requirements

The package will automatically install:
- rdkit >= 2022.9.1
- numpy >= 1.20.0
- pandas >= 1.3.0

Optional: ChemAxon `cxcalc` for protonation/tautomerization (configure path in `molecule/default_protonate.json`).
## What it does

For each input SMILES line in a `.smi` file, the program:

1. Computes a target descriptor vector (RDKit descriptors; see `molecule/descriptors.py`).
2. Runs a GA starting from a charge-specific seed pool in `input_smiles/{pos}_{neg}.smi`.
3. Scores candidates by weighted absolute error to the target descriptors.
4. Enforces charge consistency (after protonation/tautomerization).
5. Prunes/deduplicates and applies an optional diversity filter (Tanimoto cutoff) during `ga.sanitize()`.

Each input line is processed once, and input lines are parallelized.

## Requirements

- Python 3
- RDKit
- numpy
- pandas

Optional / environment-specific:
- ChemAxon `cxcalc` is used for protonation/tautomerization via `molecule/protonate.py`. Configure the executable path in `molecule/default_protonate.json`.

## Quick start

After installation, you can run the tool using:

```bash
ga-decoys \
  --input path/to/ligands.smi \
  --outdir output/ \
  --config config/test.json
```

Or if running from the repository directory:

```bash
python main.py \
  --input path/to/ligands.smi \
  --outdir output/ \
  --config config/test.json
```

Output:
- `output/results.csv` (summary of all runs)
- `output/{name}.csv` (per-input population with scores)
- `output/{name}.smi` (per-input population as SMILES file)

## Input format

`--input` should be a `.smi` file with one SMILES per line. Extra columns (e.g. names) are ignored.

## Config

The JSON config controls scoring weights and GA parameters.

Common keys:

- `population_size` (default 100)
- `mating_pool_size` (default 100)
- `generations` (default 10)
- `mutation_rate` (default 0.25)
- `n_cpus` (default 6)
- `max_molecule_size` (default 40)
- `sampled_std` (path, default `data/sampled_std.npy`) used to normalize descriptor weights

Descriptor weights (examples; 0 disables a term):
- `HeavyAtomCount`, `MolWt`, `LogP`, `NumHAcceptors`, `NumHDonors`, `TPSA`, ...

Diversity (optional):
- `tanimoto_cutoff`: if set (e.g. `0.85`), `ga.sanitize()` greedily keeps high-scoring molecules only if their max Morgan-fingerprint similarity (radius=2, nBits=1024) to **(a)** the target molecule and **(b)** already-selected molecules is below the cutoff.

## Seed pools (`input_smiles/`)

The GA starts from `input_smiles/{pos}_{neg}.smi`, where `(pos, neg)` is computed from explicit formal charges in the input SMILES.

If a given charge bucket is missing, the run will fail to initialize (no starting population). Add an appropriate seed pool file for that charge state.
