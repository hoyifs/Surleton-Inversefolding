#!/home/master/miniconda3/envs/atomsurf/bin/python

import torch
import torch.nn as nn

import esm.inverse_folding
import numpy as np
from tqdm import tqdm
from esm.inverse_folding.util import CoordBatchConverter 
import argparse

import os
print("loading models...")

model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()
encoder = model.encoder

def get_22nd_char(file_path):#get chain
    
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            if len(first_line) >= 22:
                return first_line[21]
            else:
                return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def _concatenate_coords(coords, target_chain_id, padding_length=10):

    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chain_id]]
    for chain_id in coords:
        if chain_id == target_chain_id:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated

def sample_sequence_in_complex(model, coords, target_chain_id, temperature=1.,
        padding_length=10):

    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = _concatenate_coords(coords, target_chain_id)
    device = next(model.parameters()).device

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = ['<pad>'] * all_coords.shape[0]
    for i in range(target_chain_len):
        padding_pattern[i] = '<mask>'
    sampled = model.sample(all_coords, partial_seq=padding_pattern,
            temperature=temperature, device=device)
    return sampled['encoder_out']


class EncoderOnlyModel(nn.Module):
    """
    A simplified model that only uses the encoder part of GVPTransformerModel.
    """

    def __init__(self, encoder ,dic, device = 'cuda:1'):
        super().__init__()
        self.encoder = encoder
        self.dic = dic
        self.device = device  # 记录设备信息
        self.to(self.device) 

    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.dic)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
        
        # Start with prepend token
        mask_idx = self.dic.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.dic.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i+1] = self.dic.get_idx(c)


        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        return encoder_out



def process_entry(entry_id, base_pdb_path, target_chain_id, encoder_only_model):
    pdb_path = os.path.join(base_pdb_path,f"{entry_id}.pdb")
    target_chain_id = get_22nd_char(pdb_path)
    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, [target_chain_id])

        coords, _ = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)

        sampled_seq = sample_sequence_in_complex(encoder_only_model, coords, target_chain_id, temperature=1e-6)

        flattened_seq = sampled_seq[0].squeeze(1)

        return entry_id, flattened_seq

    except Exception as e:
        print(f"Error processing {entry_id}: {e}")
        return None, None

def get_pdb_entry_ids(pdb_dir):

    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    entry_ids = [os.path.splitext(f)[0] for f in pdb_files]
    
    return entry_ids

def pdb2if1(pdb_path, out_path):
    entry_ids = get_pdb_entry_ids(pdb_path)
    total = len(entry_ids)
    errentrylist = []

    for entry_id in tqdm(entry_ids, desc="Processing PDB entries", total=total):

        out_file = os.path.join(out_path, f"{entry_id}_if1.pt")
        if os.path.exists(out_file):
            continue
        try:
            entry_id, flattened_seq = process_entry(
                entry_id, pdb_path, 'A', encoder_only_model
            )
            if entry_id is None or flattened_seq is None:
                errentrylist.append(entry_id)
                continue

            torch.save(
                flattened_seq[1:-1],
                os.path.join(out_path, f"{entry_id}_if1.pt")
            )
        except Exception as e:
            errentrylist.append(entry_id)
            print(f"[Error] {entry_id}: {e}")

    print(f"Processing completed. {len(entry_ids)-len(errentrylist):d}/{total:d} succeeded.")
    if errentrylist:
        print(f"Failed entries: {errentrylist}")

if __name__ == "__main__":
    print("create if1 emb")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, help="the directory need to preprocess of the data, the output data will be saved in the directory: ../if1_emb")
    parser.add_argument("--device", type=str, help="device string such as cuda:0 or cpu")
    args = parser.parse_args()
    parent = os.path.dirname(os.path.normpath(args.datadir))
    outdir = os.path.join(parent, "if1_emb")
    os.makedirs(outdir, exist_ok=True)
    encoder_only_model = EncoderOnlyModel(model.encoder, model.decoder.dictionary,device = args.device)
    del model
    print(args.datadir,outdir)
    pdb2if1(args.datadir,outdir)
