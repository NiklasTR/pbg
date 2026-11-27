"""Historical ROBIN atom/residue IDs in the shared ff14SB template order."""

import numpy as np

from pbg.core.constants import ONE_TO_THREE
from pbg.core.peptide import _load_ff14sb_templates

ATOM_NAMES = (
    'C CA CB CD CD1 CD2 CE CE1 CE2 CE3 CG CG1 CG2 CH2 CZ CZ2 CZ3 '
    'H HA HB HD HD1 HD2 HE HE1 HE2 HE3 HG HG1 HG2 HH HH1 HH2 HZ HZ2 HZ3 '
    'N ND1 ND2 NE NE1 NE2 NH1 NH2 NZ O OD OE OG OG1 OH OXT SD SG'
).split()
ATOM_TYPES = {name: index for index, name in enumerate(ATOM_NAMES, 1)}
AA_TYPES = {name: index for index, name in enumerate(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(), 1
)}


def encode_sequences(sequences: list[str], max_atoms: int):
    """Build ROBIN's five conditioning arrays and its atom mask.

    Use the shared ff14SB residue templates rather than a second fragment database.
    Only standard, uncapped linear L-peptides are supported.
    """
    if not sequences:
        raise ValueError('At least one sequence is required')
    templates = _load_ff14sb_templates()
    encodings = {name: np.zeros((len(sequences), max_atoms), dtype=np.int64)
                 for name in ('atom_type', 'aa_type', 'aa_pos', 'atom_pos')}
    encodings['seq_len'] = np.zeros((len(sequences), 1), dtype=np.int64)
    mask = np.zeros((len(sequences), max_atoms), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        if not sequence or any(letter not in 'ARNDCQEGHILKMFPSTWYV' for letter in sequence):
            raise ValueError(f'Unsupported ROBIN peptide sequence: {sequence!r}')
        offset = 0
        for residue_index, letter in enumerate(sequence):
            aa = ONE_TO_THREE[letter]
            if aa in ('HIE', 'HID', 'HIP'):
                aa = 'HIS'
            template = 'HIE' if aa == 'HIS' else aa
            prefix = 'C' if residue_index == len(sequence) - 1 else ('N' if residue_index == 0 else '')
            names = templates[prefix + template].atom_names
            if offset + len(names) > max_atoms:
                raise ValueError(f'{sequence} exceeds max_num_atoms={max_atoms}')
            for name in names:
                if name.startswith('H') and name[-1] in '123':
                    if not (aa in ('HIS', 'PHE', 'TRP', 'TYR') and name[:2] in ('HE', 'HD', 'HZ', 'HH')):
                        name = name[:-1]
                if name[:2] in ('OE', 'OD'):
                    name = name[:-1]
                encodings['atom_type'][row, offset] = ATOM_TYPES.get(name, 0)
                encodings['aa_type'][row, offset] = AA_TYPES[aa]
                encodings['aa_pos'][row, offset] = residue_index + 1
                offset += 1
        encodings['atom_pos'][row, :offset] = np.arange(offset)
        encodings['seq_len'][row, 0] = len(sequence)
        mask[row, :offset] = 1
    return encodings, mask
