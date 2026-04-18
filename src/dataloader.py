
import os
import sys
import time
import torch
import numpy as np
from Bio import PDB
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))
from atomsurf.utils.data_utils import AtomBatch
from atomsurf.utils.wrappers import DefaultLoader
from torch.utils.data import IterableDataset

def get_pdb_sequence(pdb_file_path):

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    ppb = PDB.PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()
        sequences.append(str(sequence))

    return ''.join(sequences)

class Proteinsingleset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))  


    def __len__(self):
        return len(self.files)
    
    def getname(self):
        return self.files

    def __getitem__(self, idx):

        path = os.path.join(self.data_dir, self.files[idx])
        item = torch.load(path)
        return item['pdata'], item['seq'], self.files[idx]

    


class ProteinDataset(IterableDataset):
    
    def __init__(self, pdb_dir, esm_dir, surface_dir, rgraph_dir, surface_bb = False):

        default_loader = DefaultLoader(surface_dir=surface_dir, graph_dir=rgraph_dir, embeddings_dir=esm_dir)

        pfiles = os.listdir(pdb_dir)

        pro_name = [f.split('.')[0] for f in pfiles if f.endswith('.pdb')]

        self.proseq = []
        self.sglist = []

        count = 0
        for i in pro_name:
            if count % 100 == 0:
                print(f"Loading {count}th protein data")

            try:
                s, g = default_loader(i)
                prosg = Data(surface=s, graph=g)
                seq = get_pdb_sequence(os.path.join(pdb_dir, f"{i}.pdb"))
                self.proseq.append([i, seq])
                self.sglist.append(prosg)
            except Exception as e:
                print(f"[WARNING] Failed to load {i}: {e}")

            count += 1
        print("Done with loading pdb file")

    def __len__(self):
        return len(self.proseq)

    def __getitem__(self, idx):
        pdata = self.sglist[idx]
        _, pseq = self.proseq[idx]
        return pdata, pseq

    def getname(self):
        return self.proseq



def seq_to_onehot(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    aa_dict = {amino_acids[i]: i for i in range(len(amino_acids))}

    onehot_matrix = np.zeros((len(seq), len(amino_acids)), dtype=float)
    
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            onehot_matrix[i, aa_dict[aa]] = 1.0
    return onehot_matrix

def _onehot_to_seq(onehot_matrix):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq = [amino_acids[np.argmax(onehot)] for onehot in onehot_matrix]
    return ''.join(seq)


def collate_fn(batch):
    inputs, labels, name = zip(*batch)

    asbatch = AtomBatch.from_data_list(inputs)

    concat_seq = "".join(labels)

    concat_seq_onehot = seq_to_onehot(concat_seq)

    concat_seq_tensor = torch.from_numpy(concat_seq_onehot).float()

    return asbatch, concat_seq_tensor



