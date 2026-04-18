
import os
from Bio.PDB import PDBParser, PDBIO, Select

input_folder = "/data3/ai_hyff/scopedata/pdb" 
output_folder = "/data3/ai_hyff/scopedata/pdb_chain"

os.makedirs(output_folder, exist_ok=True)


parser = PDBParser(QUIET=True)
io = PDBIO()

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id
c = 0
for pdb_file in os.listdir(input_folder):
    c += 1
    if c%100 == 0:
        print(c)
    if pdb_file.endswith(".pdb"):
        file_path = os.path.join(input_folder, pdb_file)

        structure = parser.get_structure(pdb_file, file_path)
        

        for model in structure:
            for chain in model:
                chain_id = chain.get_id()  
                chain_filename = f"{pdb_file[:4]}_{chain_id}.pdb"  
                chain_filepath = os.path.join(output_folder, chain_filename)
                
                # 创建选择器并保存链
                select = ChainSelect(chain_id)
                io.set_structure(structure)
                io.save(chain_filepath, select=select)
                