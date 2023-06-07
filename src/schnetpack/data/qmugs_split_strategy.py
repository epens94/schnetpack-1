from typing import List
import operator
import torch
import logging
import numpy as np
from schnetpack.data import SplittingStrategy
from tqdm import tqdm
import itertools
import random
__all__ = ["QMugsSplit"]



class QMugsSplit(SplittingStrategy):

    def __init__(self):
        super().__init__()

    
    def split(self,dataset,*split_sizes) -> List[torch.tensor]:

        n_excluded = None
        num_train, num_val, num_test = split_sizes
        total = num_train+num_test+num_val
        
        vals = dataset.metadata["QMugs"]

        #d1 = {
        #    dataset[i]["_idx"].detach().numpy()[0]:[i,
        #        dataset[i]["_idx"].detach().numpy()[0],int(dataset[i]["CHEMBL_ID"].detach().numpy()[0])
        #    ] for i,n in tqdm(enumerate(dataset))
        #}
        #val = np.array([d1[key] for key in list(d1.keys())])[:,1:]
        d1 = {str(vals[i][0]): [vals[i][0], vals[i][0],vals[i][1]] for i in range(len(vals))}

        val = np.array([d1[key] for key in list(d1.keys())])[:,1:]
        df = val[:n_excluded]

        logging.info(f"Creating database idx to chemblid map ...")
        idx_to_chemblid_map = {
            chembl_id : df[df[:,1] == chembl_id ][:,0].tolist() for chembl_id  in np.unique(df[:,1]) 
        }

        train_idx, val_idx, test_idx = ([],[],[])
        # percentage of split sizes are needed
        num_train,num_val,num_test = [x / total for x in [num_train, num_val, num_test]]

        # creates filtered idx_to_chemblid_map dict for specific n of confs e.g there are 862 entries where only 1 conformer exists
        for n_conf in [1,2,3]:
            conf_dict = {
                chemblid: idx_to_chemblid_map[chemblid] for chemblid in [
                    chemblid for chemblid in tqdm(list(idx_to_chemblid_map.keys())) if len(idx_to_chemblid_map[chemblid]) == n_conf
                    ] 
                }

            logging.info("Collecting all database indices for all CHEMBL_ID with only {} conformer".format(n_conf))
            indices = np.random.permutation(list(conf_dict.keys()))
            lengths = [int(num_train*len(conf_dict)),
                int(num_val*len(conf_dict)),
                len(conf_dict) - int(num_train*len(conf_dict)) - int(num_val*len(conf_dict))
                ]
            offsets = torch.cumsum(torch.tensor(lengths), dim=0)

            train,val,test = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, lengths)
            ]

            train,val,test = [
            list(itertools.chain.from_iterable([conf_dict[_idx] for _idx in LIST])) for LIST in [train,val,test]]
                
            train_idx.extend(train)
            val_idx.extend(val)
            test_idx.extend(test)
        
        train_idx, val_idx, test_idx = [
            np.random.permutation(_idxs).tolist() for _idxs in [train_idx, val_idx, test_idx]
            ]
        
        train_idx = random.sample([d1[n][0] for n in train_idx ],split_sizes[0])
        val_idx = random.sample([d1[n][0] for n in val_idx ],split_sizes[1])
        test_idx = random.sample([d1[n][0] for n in test_idx],split_sizes[2])

        return [train_idx, val_idx, test_idx]




