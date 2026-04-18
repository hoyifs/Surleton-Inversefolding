#!/home/master/miniconda3/envs/atomsurf/bin/python

import os
import sys
import argparse
from tqdm import tqdm
import argparse

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))

from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, help="the directory need to preprocess of the data, the output data will be saved in this directory")
    parser.add_argument("--del_mode", action="store_true", help="if set, the files with HETATM or out of range length will be deleted")
    parser.add_argument("--preprocess", action="store_true", help="if set, the data will be preprocessed")
    args = parser.parse_args()

    del_mode = args.del_mode
    prepocess = args.preprocess
    datadir = args.datadir
    print("all the files processed in the file\n" + datadir)

    files_with_hetatm = []
    files_lenth_out_of_range = []
    for filename in os.listdir(datadir+"/pdb/"):

        if os.path.isfile(os.path.join(datadir+"/pdb/", filename)) and filename.endswith('.pdb'):
            with open(os.path.join(datadir+"/pdb/", filename), 'r') as file:
                lines = file.readlines()
                for i,line in enumerate(lines):
                    if i == 0:
                        start = int(line[22:26])
                    if line.startswith("HETATM"):
                        files_with_hetatm.append(filename)
                        break 
                    if i == len(lines) - 1:
                        if int(line[22:26])-start+1 >= 3000 or int(line[22:26])-start+1 < 20:
                            files_lenth_out_of_range.append(filename)

    print(f"Files with HETATM:{len(files_with_hetatm)}")
    print(f"Files with length out of range:{len(files_lenth_out_of_range)}")
    all_files = set(files_with_hetatm) | set(files_lenth_out_of_range)
    print(f"Total files to be deleted:{len(all_files)}")


    if del_mode:

        for filename in all_files:
            file_path = os.path.join(datadir+"/pdb/", filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    if prepocess:
        dataset = PreprocessDataset(data_dir= datadir, radius=1.4)
        do_all(dataset, num_workers=20)

