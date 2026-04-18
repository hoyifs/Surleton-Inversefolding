# Surleton-Inversefolding

Surleton is a **surface-aware inverse protein folding framework** that jointly models protein backbone geometry and molecular surface structure.  
By integrating surface-level geometric representations with backbone-conditioned modeling, Surleton improves sequence recovery and confidence, particularly for **surface-exposed residues** that are underconstrained by backbone-only inverse folding methods.

##Installation

Create a new environment named surleton
```bash
cd Surleton
conda create -n surleton -y
conda activate surleton
conda install python=3.8
```
Install pytorch (GPU/CPU)
```bash
###GPU
conda install cudatoolkit=11.7 -c nvidia
conda install pytorch=1.13 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.3.0 pytorch-scatter pytorch-sparse pytorch-spline-conv pytorch-cluster -c pyg
pip install pyg-lib==0.4.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
###CPU
pip install --no-cache-dir torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir torch_scatter torch_sparse torch_spline_conv torch_cluster pyg_lib -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install --no-cache-dir torch_geometric==2.3.0 -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
```
If PyTorch and MKL libraries are incompatible on your system, run:
```bash
conda install -n surleton mkl=2024.0.0
```

install diffusion-net and other third-party packages
```bash
pip install git+https://github.com/pvnieo/diffusion-net-plus.git
pip install -r ./bin/requirements.txt
```
Alternatively, if GitHub access is restricted:
```
pip install git+https://gitclone.com/github.com/pvnieo/diffusion-net-plus.git
pip install -r ./bin/requirements.txt
```
##Usage

Example data adapted from published literature are provided under the Surleton/example directory. These files demonstrate a complete inverse protein folding workflow using Surleton.

To apply Surleton to other proteins, users should prepare a PDB file containing protein atoms only. All non-protein entries, including ligands, ions, water molecules, and heteroatoms, should be removed so that only ATOM records corresponding to the protein remain. The processed PDB file should then be placed in the Surleton/example/pdb directory.

After preparing the PDB file, run the preprocessing pipeline to construct backbone and surface representations. Upon completion, the processed dataset will be generated in the Surleton/example/if1_pt_dataset directory. The resulting files are stored in .pt format and can be used directly as model input.

The Surleton/src directory contains example Jupyter notebooks (e.g., run.ipynb) demonstrating how to perform sequence generation using pretrained Surleton models. By executing the notebook cells sequentially, users can generate amino acid sequences for the corresponding protein structures.