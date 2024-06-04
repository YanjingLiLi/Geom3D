import os.path as osp
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.preprocessing import normalize
import h5py
import lmdb
import pickle as pkl
import json
import msgpack
import pandas as pd
import scipy
import io
import gzip
import logging
from Bio.PDB.Polypeptide import three_to_one, is_aa
from tqdm import tqdm

import torch, math
import torch.nn.functional as F
import torch_cluster

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset


class DatasetPSR(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split='train'):
        self.split = split
        self.root = root
        self.device = "cuda"
        self.index_columns = ['ensemble', 'subunit', 'structure', 'model', 'chain', 'residue']
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, "X":20}

        super(DatasetPSR, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed_PSR_label'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def deserialize(self, x, serialization_format):
        """
        Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
        """
        if serialization_format == 'pkl':
            return pkl.loads(x)
        elif serialization_format == 'json':
            serialized = json.loads(x)
        elif serialization_format == 'msgpack':
            serialized = msgpack.unpackb(x)
        else:
            raise RuntimeError('Invalid serialization format')
        
        return serialized
        
    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))
    
    def get_side_chain_angle_encoding(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.diherals_ProNet(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.diherals_ProNet(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.diherals_ProNet(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.diherals_ProNet(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.diherals_ProNet(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs
    
    def get_backbone_angle_encoding(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.diherals_ProNet(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    def diherals_ProNet(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))

        return torsion

    def _three_to_one(self, residue):
        try:
            return three_to_one(residue)
        except KeyError:
            return "X"

    def parse_protein_df(self, protein_df):
        atom_names, atom_pos, residue_type, atom_amino_id = [], [], [], []
        all_residues = protein_df["residue"].unique()
        
        residue_num = 0 
        invalid = False
        for residue in all_residues:
            residue_name = protein_df[protein_df["residue"] == residue]["resname"].iloc[0]
            if is_aa(residue_name) or residue_name == "UNK":
                residue_df = protein_df[protein_df["residue"] == residue]
                if residue_df["fullname"].str.strip().isin(["N"]).any() and residue_df["fullname"].str.strip().isin(["CA"]).any() and residue_df["fullname"].str.strip().isin(["C"]).any():
                    residue_id = self.letter_to_num[self._three_to_one(residue_name)]
                    residue_type.append(residue_id)
                    for index, row in residue_df.iterrows():
                        atom_names.append(row["fullname"].strip())
                        if [row["x"], row["y"], row["z"]] == [0., 0., 0.]:
                            invalid = True
                        atom_pos.append([row["x"], row["y"], row["z"]])
                        atom_amino_id.append(residue_num)
                    
                    residue_num += 1
                        
        if invalid:
            return None, None, None, None
        
        return atom_names, np.array(atom_pos), residue_type, np.array(atom_amino_id)
    
    def get_key_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, 'N')
        mask_ca = np.char.equal(atom_names, 'CA')
        mask_c = np.char.equal(atom_names, 'C')
        mask_cb = np.char.equal(atom_names, 'CB')
        mask_g = np.char.equal(atom_names, 'CG') | np.char.equal(atom_names, 'SG') | np.char.equal(atom_names, 'OG') | np.char.equal(atom_names, 'CG1') | np.char.equal(atom_names, 'OG1')
        mask_d = np.char.equal(atom_names, 'CD') | np.char.equal(atom_names, 'SD') | np.char.equal(atom_names, 'CD1') | np.char.equal(atom_names, 'OD1') | np.char.equal(atom_names, 'ND1')
        mask_e = np.char.equal(atom_names, 'CE') | np.char.equal(atom_names, 'NE') | np.char.equal(atom_names, 'OE1')
        mask_z = np.char.equal(atom_names, 'CZ') | np.char.equal(atom_names, 'NZ')
        mask_h = np.char.equal(atom_names, 'NH1')
        mask_u = np.char.equal(atom_names, 'U')

        pos_n = np.full((len(amino_types),3),np.nan)
        pos_n[atom_amino_id[mask_n]] = atom_pos[mask_n]
        pos_n[atom_amino_id[mask_u]] = atom_pos[mask_u]
        pos_n = torch.FloatTensor(pos_n)

        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca[atom_amino_id[mask_u]] = atom_pos[mask_u]
        pos_ca = torch.FloatTensor(pos_ca)

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c[atom_amino_id[mask_u]] = atom_pos[mask_u]
        pos_c = torch.FloatTensor(pos_c)

        # if data only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h
                
    def extract_protein_data(self, protein_df):
        data = Data()
    
        atom_names, atom_pos, residue_types, atom_amino_id = self.parse_protein_df(protein_df)

        if atom_names == None:
            return None

        ##### compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1 #####
        # extract key atom (e.g., backbone) positions
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h = self.get_key_atom_pos(residue_types, atom_names, atom_amino_id, atom_pos)

        # calculate side chain torsion angles, up to four
        # do encoding
        side_chain_angle_encoding = self.get_side_chain_angle_encoding(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_angle_encoding[torch.isnan(side_chain_angle_encoding)] = 0

        # three backbone torsion angles
        backbone_angle_encoding = self.get_backbone_angle_encoding(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        backbone_angle_encoding[torch.isnan(backbone_angle_encoding)] = 0

        data.seq = torch.LongTensor(residue_types)
        data.side_chain_angle_encoding = side_chain_angle_encoding
        data.backbone_angle_encoding = backbone_angle_encoding
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        data.x = atom_names
        data.pos = torch.tensor(atom_pos)
        data.num_nodes = len(pos_ca)

        return data

    def process(self):  
        print('Beginning Processing ...')

        data_list = []

        env = lmdb.open(osp.join(self.root, self.split), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
            self._id_to_idx = self.deserialize(
                txn.get(b'id_to_idx'), self._serialization_format)

        self._env = env
        
        for index in tqdm(range(self._num_examples), desc="all samples"):
            with self._env.begin(write=False) as txn:
                compressed = txn.get(str(index).encode())
                buf = io.BytesIO(compressed)
                with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                    serialized = f.read()
                try:
                    item = self.deserialize(serialized, self._serialization_format)
                except:
                    return None
                
                # Recover special data types (currently only pandas dataframes).
                if 'types' in item.keys():
                    for x in item.keys():
                        if (self._serialization_format=='json') and (item['types'][x] == str(pd.DataFrame)):
                            item[x] = pd.DataFrame(**item[x])
                else:
                    logging.warning('Data types in item %i not defined. Will use basic types only.'%index)

                if 'file_path' not in item:
                    item['file_path'] = str(self.data_file)
                if 'id' not in item:
                    item['id'] = str(index)

                protein = item["atoms"]

                data = self.extract_protein_data(protein)

                if data != None:
                    for i in range(len(data.coords_ca)):
                        for j in range(i+1, len(data.coords_ca)):
                            if torch.equal(data.coords_ca[i], data.coords_ca[j]):
                                data = None
                                break
                        if data == None:
                            break

                if data != None:
                    data.y = item["scores"]["gdt_ts"]
                    data.id = item['id']
                    data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')      

if __name__ == "__main__":
    for split in ["test"]:
    #for split in ['validation']:
        print('#### Now processing {} data ####'.format(split))
        dataset = DatasetPSR(root="/lustre07/scratch/liusheng/atom3d_data/PSR/split-by-year/data", split=split)