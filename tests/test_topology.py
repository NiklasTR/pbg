from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from pbg.core.frame import DataFrame
from pbg.core.helm import (
    load_helm_monomer_library,
    mol_from_helm,
    rewrite_helm_residues,
    rotate_cyclic_helm,
)
from pbg.core.peptide import sequence_to_helm, sequence_to_smiles
from pbg.core.tokeniser import load_tokeniser_default
from pbg.core.topology import ATOM_ORDER_IMPLEMENTATION, PromptTopology
from pbg.models.pbg.config import PBGConfig
from pbg.models.pbg.loader import augmenter_from_config, prepare_frame

CYCLE = 'PEPTIDE1{A.L.[dL].P.[dL].F}$PEPTIDE1,PEPTIDE1,1:R1-6:R2$$$'


def source_frame(helm, mol):
    mol = Chem.AddHs(mol)
    for i, atom in enumerate(mol.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    smiles = Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True, isomericSmiles=True)
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    xyz = np.arange(len(z) * 3, dtype=np.float32).reshape(-1, 3) * .001
    return replace(DataFrame.from_smiles(smiles), helm=helm, temperature=300., environment='vacuum',
        conformer=np.column_stack((z, xyz)).astype(np.float32), metadata={'pilot_key': 'example'})


def test_cycle_rotations_and_rewrites_have_identical_topology_traversal_and_targets():
    library = load_helm_monomer_library()
    frame = source_frame(CYCLE, mol_from_helm(CYCLE, library))
    cfg = PBGConfig(vocab_name='joint_bin4096', atom_order='prompt_canonical_v1',
        prompt_format='compact_kelvin_v1', center=False, augment=False, translate=False)
    vocab = load_tokeniser_default(vocab_name='joint_bin4096')
    expected = PromptTopology.compile(CYCLE)
    compiled, token_arrays = [], []
    for offset in range(6):
        helm = rotate_cyclic_helm(CYCLE, offset, library)
        for variant in (helm, rewrite_helm_residues(helm, [1, 4], library)):
            topology = PromptTopology.compile(variant)
            assert topology == expected
            compiled.append(topology.source_rows(frame.smiles))
            prepared = prepare_frame(replace(frame, helm=variant), cfg, vocab,
                augmenter_from_config(cfg), rng=np.random.default_rng(42))
            assert prepared.smiles == expected.mapped_smiles
            token_arrays.append(prepared.coord_ids)
    assert all(rows == compiled[0] for rows in compiled)
    for tokens in token_arrays[1:]:
        np.testing.assert_array_equal(tokens, token_arrays[0])


def test_chem_branch_traversals_permute_coordinates_and_preserve_inverse_mapping():
    frame = source_frame('CHEM1{[CCCCCCOC(C)=O]}$$$$', Chem.MolFromSmiles('CCCCCCOC(=O)C'))
    topology = PromptTopology.compile(frame.helm)
    alternative = PromptTopology.compile('CHEM1{[CC(=O)OCCCCCC]}$$$$')
    assert topology == alternative
    rows = topology.source_rows(frame.smiles)
    assert rows[:10] == (0, 1, 2, 3, 4, 5, 6, 7, 9, 8)
    aligned = topology.align_frame(frame)
    np.testing.assert_array_equal(aligned.conformer, frame.conformer[list(rows)])
    np.testing.assert_array_equal(aligned.conformer[np.argsort(rows)], frame.conformer)
    assert topology.agrees_with(aligned.smiles)
    again = topology.align_frame(aligned)
    np.testing.assert_array_equal(again.conformer, aligned.conformer)
    assert again.metadata['source_atom_rows'] == list(rows)
    assert again.metadata['source_mapped_smiles'] == frame.smiles


def test_canonical_solute_keeps_water_blocks_and_labels():
    solute = Chem.AddHs(Chem.MolFromSmiles('OC'))
    waters = Chem.AddHs(Chem.MolFromSmiles('O.O'))
    # Source shell is O/H/H per molecule, rather than AddHs' O/O/H/H/H/H.
    waters = Chem.RenumberAtoms(waters, (0, 2, 3, 1, 4, 5))
    frame = source_frame('CHEM1{[CO]}$$$$', Chem.CombineMols(solute, waters))
    topology = PromptTopology.compile(frame.helm, '2 water, gaff2-am1bcc-tip3p-rf2nm')
    aligned = topology.align_frame(frame)
    assert topology.n_solute_atoms == 6
    assert topology.n_waters == 2
    assert topology.source_rows(frame.smiles)[6:] == tuple(range(6, 12))
    np.testing.assert_array_equal(aligned.conformer[6:], frame.conformer[6:])
    assert [a[0] for a in topology.atoms[6:]] == [8, 1, 1, 8, 1, 1]


def test_graph_and_stereo_mismatches_fail_before_coordinate_alignment():
    frame = source_frame('CHEM1{[F[C@H](Cl)Br]}$$$$', Chem.MolFromSmiles('F[C@@H](Cl)Br'))
    with pytest.raises(ValueError, match='chemistry or stereochemistry'):
        PromptTopology.compile(frame.helm).align_frame(frame)
    bad_z = replace(frame, helm='CHEM1{[F[C@@H](Cl)Br]}$$$$', conformer=frame.conformer.copy())
    bad_z.conformer[0, 0] = 8
    with pytest.raises(ValueError, match='atomic numbers'):
        PromptTopology.compile(bad_z.helm).align_frame(bad_z)


def test_isotopic_hydrogens_remain_in_graph_and_mapping():
    frame = source_frame('CHEM1{[OC[2H]]}$$$$', Chem.MolFromSmiles('[2H]CO'))
    topology = PromptTopology.compile(frame.helm)
    aligned = topology.align_frame(frame)
    assert topology.agrees_with(aligned.smiles)
    assert sum(a[2] == 2 for a in topology.atoms) == 1


def test_imidazole_mapping_distinguishes_n_from_nh_in_real_source_graph():
    # Real joint-loader failure: the heavy graph alone admits an N/[nH] swap.
    smiles = '[c:1]1([H:6])[c:2]([H:7])[n:3][c:4]([H:8])[n:5]1[H:9]'
    topology = PromptTopology.compile('CHEM1{[c1c[nH]cn1]}$$$$')
    rows = topology.source_rows(smiles)
    assert rows == (1, 0, 4, 3, 2, 6, 5, 8, 7)
    frame = replace(DataFrame.from_smiles(smiles),
        conformer=np.column_stack(([6, 6, 7, 6, 7, 1, 1, 1, 1],
            np.arange(27).reshape(9, 3))).astype(np.float32))
    aligned = topology.align_frame(frame)
    assert topology.agrees_with(aligned.smiles)
    np.testing.assert_array_equal(aligned.conformer[np.argsort(rows)], frame.conformer)


def test_sampling_compiles_from_prompt_and_rejects_source_order_for_canonical_model():
    from pbg.models.pbg.system import MolecularSystem, prompt_arrays
    helm = 'CHEM1{[CCCCCCOC(C)=O]}$$$$'
    system = MolecularSystem.from_protocol(id='hexyl-acetate', helm=helm,
        target_id='gaff2-am1bcc', temperature_K=300.)
    cfg = PBGConfig(vocab_name='joint_bin4096', atom_order='prompt_canonical_v1')
    model = SimpleNamespace(config=cfg, vocab=load_tokeniser_default(vocab_name='joint_bin4096'))
    *_, z = prompt_arrays(model, system, 1)
    assert tuple(z.astype(int)) == tuple(a[0] for a in PromptTopology.compile(helm).atoms)
    source = source_frame(helm, Chem.MolFromSmiles('CCCCCCOC(=O)C'))
    with pytest.raises(ValueError, match='atom traversal'):
        prompt_arrays(model, replace(system, mapped_smiles=source.smiles), 1)


def test_config_freezes_compiler_and_rejects_extra_atom_permutations():
    cfg = PBGConfig(atom_order='prompt_canonical_v1')
    assert cfg.atom_order_implementation == ATOM_ORDER_IMPLEMENTATION
    with pytest.raises(ValueError, match='implementation differs'):
        replace(cfg, atom_order_implementation='different-compiler')
    with pytest.raises(ValueError, match='unknown atom order'):
        PBGConfig(atom_order='prompt_residue_canonical_v1')
    with pytest.raises(ValueError, match='Unknown prompt atom order'):
        PromptTopology.compile('CHEM1{[CO]}$$$$', order='source_v1')
    assert PromptTopology.applies_to('PEPTIDE1{A.G}$$$$')


@pytest.mark.parametrize('symbol', list('ARNDCQEGHILKMFPSTWYV'))
def test_linear_named_and_inline_monomers_use_one_contract(symbol):
    library = load_helm_monomer_library()
    for sequence in (symbol, symbol + 'G', 'G' + symbol, 'G' + symbol + 'G'):
        helm = sequence_to_helm(sequence)
        expected = PromptTopology.compile(helm)
        assert expected.mapped_smiles == sequence_to_smiles(sequence)
        rewritten = rewrite_helm_residues(helm, list(range(1, len(sequence) + 1)), library)
        assert PromptTopology.compile(rewritten) is expected


def test_linear_compiler_drives_training_and_sampling_with_identical_traversal():
    from pbg.core.molecule import atom_numbers_from_smiles
    from pbg.models.pbg.system import MolecularSystem, prompt_arrays
    helm = sequence_to_helm('ARKHDEP')
    topology = PromptTopology.compile(helm, '2 water, ff14sb-tip3p-rf2nm')
    rewritten = rewrite_helm_residues(helm, [1, 4, 7], load_helm_monomer_library())
    assert PromptTopology.compile(rewritten, '2 water, ff14sb-tip3p-rf2nm') == topology
    z = atom_numbers_from_smiles(topology.mapped_smiles)
    frame = replace(DataFrame.from_smiles(topology.mapped_smiles), helm=rewritten,
        environment='2 water, ff14sb-tip3p-rf2nm', temperature=300.,
        conformer=np.column_stack((z, np.zeros((len(z), 3)))).astype(np.float32))
    cfg = PBGConfig(vocab_name='joint_bin4096', atom_order='prompt_canonical_v1',
        augment=False, translate=False, center=False)
    vocab = load_tokeniser_default(vocab_name='joint_bin4096')
    prepared = prepare_frame(frame, cfg, vocab, augmenter_from_config(cfg), rng=np.random.default_rng(0))
    assert prepared.smiles == topology.mapped_smiles
    assert prepared.metadata['source_atom_rows'] == list(range(len(z)))
    system = MolecularSystem.from_protocol(id='linear', helm=rewritten, target_id='ff14sb-tip3p-rf2nm',
        temperature_K=300., n_waters=2)
    *_, sampled_z = prompt_arrays(SimpleNamespace(config=cfg, vocab=vocab), system, 1)
    np.testing.assert_array_equal(sampled_z, z)


def test_linear_unregistered_monomer_is_not_silently_changed():
    # A valid residue outside ff14SB keeps the chemistry its token states (see
    # test_linear_peptides_with_residues_outside_ff14sb_keep_ff14sb_states); a
    # token that cannot join the chain still fails.
    with pytest.raises(ValueError, match='registered ff14SB|Unsupported linear|does not have R1'):
        PromptTopology.compile('PEPTIDE1{A.[NCC(C)C(=O)[*:2]]}$$$$')


def test_fixed_solute_sampling_rejects_source_order_before_model_execution(monkeypatch):
    import importlib
    sample = importlib.import_module('pbg.models.pbg.sample')
    helm = 'CHEM1{[CCCCCCOC(C)=O]}$$$$'
    frame = source_frame(helm, Chem.MolFromSmiles('CCCCCCOC(=O)C'))
    system = sample.MolecularSystem.from_protocol(id='shell', helm=helm, target_id='gaff2-am1bcc-tip3p-rf2nm',
        temperature_K=300., n_waters=1)
    model = SimpleNamespace(config=PBGConfig(atom_order='prompt_canonical_v1'))
    monkeypatch.setattr(sample, 'inference_model', lambda model: model)
    with pytest.raises(ValueError, match='Fixed solute'):
        sample.sample_waters_given_solute(model, system, None, solute_frames=frame.conformer[None],
            n_solute_atoms=len(frame.conformer), solute_smiles=frame.smiles)


def test_prompts_rotate_cycles_and_rewrite_up_to_two_residues():
    from pbg.data.joint import helm_templates, select_representation
    library = load_helm_monomer_library()
    stored = rewrite_helm_residues(CYCLE, list(range(1, 7)), library)  # many_cycles stores this form
    linear = sequence_to_helm('ARKHDEP')
    assert helm_templates(stored)[2] == helm_templates(stored)[3]
    cycle_frame = source_frame(stored, mol_from_helm(CYCLE, library))
    linear_frame = replace(cycle_frame, helm=linear)
    for frame, n, reference in ((cycle_frame, 6, CYCLE), (linear_frame, 7, linear)):
        topology = PromptTopology.compile(reference)
        counts, rotations = set(), set()
        for key in range(200):
            keyed = replace(frame, metadata={'pilot_key': f'peptide-{key}'})
            selected, _ = select_representation(keyed, 42)
            assert selected.metadata['helm_rewrite_changed'] <= selected.metadata['helm_rewrite_selected']
            counts.add(selected.metadata['helm_rewrite_selected'])
            rotations.add(selected.metadata['helm_rotation'])
            prompt = PromptTopology.compile(selected.helm)
            assert prompt.canonical_smiles == topology.canonical_smiles
            if n == 7:
                assert prompt == topology
        assert counts == {0, 1, 2}
        assert rotations == (set(range(n)) if n == 6 else {0})


def test_sampling_policy_checks_systems_against_the_checkpoint_atom_order():
    from pbg.models.pbg.system import MolecularSystem, SamplingPolicy
    vocab = load_tokeniser_default(vocab_name='joint_bin4096')
    config = PBGConfig(vocab_name='joint_bin4096', prompt_format='compact_kelvin_v1', atom_order='prompt_canonical_v1')
    policy = SamplingPolicy.from_model(SimpleNamespace(config=config, vocab=vocab))
    assert policy.atom_order == 'prompt_canonical_v1'
    helm = 'CHEM1{[CCCCCCOC(C)=O]}$$$$'
    system = MolecularSystem.from_protocol(id='p', helm=helm, target_id='gaff2-am1bcc', temperature_K=300.)
    assert policy.system(system) == system
    source = source_frame(helm, Chem.MolFromSmiles('CCCCCCOC(=O)C'))
    with pytest.raises(ValueError, match='prompt-derived atom traversal'):
        policy.system(replace(system, mapped_smiles=source.smiles))


def test_linear_peptides_with_residues_outside_ff14sb_keep_ff14sb_states():
    flat = lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    d_leu, abu = '[CC(C)C[C@@H](N[*:1])C(=O)[*:2]]', '[CC[C@@H](N[*:1])C(=O)[*:2]]'
    l_peptide = PromptTopology.compile(sequence_to_helm('KLWN'), 'vacuum, ff14sb')
    named = PromptTopology.compile('PEPTIDE1{K.[dL].W.N}$$$$', 'vacuum, ff14sb')
    for helm in (f'PEPTIDE1{{K.{d_leu}.W.N}}$$$$', 'PEPTIDE1{K.[dL].W.N}$$$$', f'PEPTIDE1{{A.{abu}.A}}$$$$'):
        topology = PromptTopology.compile(helm, 'vacuum, ff14sb')
        charges = sorted(a.GetFormalCharge() for a in Chem.MolFromSmiles(topology.canonical_smiles).GetAtoms()
                         if a.GetFormalCharge())
        if 'K' in helm:
            # Same graph and ff14SB states (NH3+ terminus and Lys, COO-) as the L-peptide; D at residue 2.
            assert topology == named
            assert flat(topology.canonical_smiles) == flat(l_peptide.canonical_smiles)
            assert topology.canonical_smiles != l_peptide.canonical_smiles and charges == [-1, 1, 1]
        else:
            assert charges == [-1, 1]
