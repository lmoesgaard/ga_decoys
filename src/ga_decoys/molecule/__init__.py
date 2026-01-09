from .util import MoleculeOptions
from .util import mol_ok, ring_ok
from .descriptors import descriptor_list, get_actual_formal_charge, get_prop_arr


def __getattr__(name: str):
	# Avoid importing .protonate at module import time so that
	# `python -m ga_decoys.molecule.protonate` doesn't warn about the
	# module being in sys.modules prior to execution.
	if name == "protonate_smiles":
		from .protonate import protonate_smiles

		return protonate_smiles
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
	"MoleculeOptions",
	"mol_ok",
	"ring_ok",
	"descriptor_list",
	"get_actual_formal_charge",
	"get_prop_arr",
	"protonate_smiles",
]