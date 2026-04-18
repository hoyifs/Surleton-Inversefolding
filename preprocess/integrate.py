#!/home/master/miniconda3/envs/atomsurf/bin/python
 
import os
import torch
import argparse
import sys
from tqdm import tqdm
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import ProteinDataset


def save_proteins_as_single_files_by_name(pdb_dir, esm_dir, surface_dir, rgraph_dir, save_dir):
    dataset = ProteinDataset(pdb_dir, esm_dir, surface_dir, rgraph_dir)
    os.makedirs(save_dir, exist_ok=True)
    c = 0
    for idx in tqdm(range(len(dataset)), desc="Saving proteins"):
        pdata = dataset.sglist[idx]
        pname, seq = dataset.proseq[idx]
        if (hasattr(pdata,"surface") and hasattr(pdata,"graph")):
            c += 1
            save_path = os.path.join(save_dir, f"{pname}.pt")
            torch.save({'pdata': pdata, 'seq': seq}, save_path)
        else:
            print(pname+"  surface:"+str(hasattr(pdata,"surface"))+"  graph:"+str(hasattr(pdata,"graph")))

    print(f"Saved {c} samples into {save_dir}")



def main(datadir, bbf, ifpt):

    pdb_dir = os.path.join(datadir, "pdb")
    surface_dir = os.path.join(datadir, bbf)
    rgraph_dir = os.path.join(datadir, "rgraph")
    esm_dir = os.path.join(datadir, "if1_emb")###
    pt_dataset_dir = os.path.join(datadir, ifpt)
    print("pdb_dir:"+pdb_dir)
    print("esm_dir:"+esm_dir)
    print("surface_dir:"+surface_dir)
    print("rgraph_dir:"+rgraph_dir)
    print("pt_dataset_dir:"+pt_dataset_dir)


    save_proteins_as_single_files_by_name(pdb_dir, esm_dir, surface_dir, rgraph_dir, pt_dataset_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, help="the directory need to preprocess of the data")
    parser.add_argument("--bb", action="store_true", help="backbone channel")
    args = parser.parse_args()
    datadir = args.datadir
    print("all the files processed in the dir:\n" + datadir)

    if args.bb:
        bbf = "surfaces_3_bb"
        ifpt = "if1_pt_dataset_3_bb"
    else:
        bbf = "surfaces_0.1"
        ifpt = "if1_pt_dataset"

    pt_dataset_dir = os.path.join(datadir, ifpt)

    if not os.path.exists(pt_dataset_dir):
        os.makedirs(pt_dataset_dir)

    main(datadir, bbf, ifpt)
