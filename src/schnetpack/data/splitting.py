from typing import Optional, List, Dict, Tuple, Union
import math
import torch
import numpy as np
import scipy.sparse as sp

__all__ = [
    "SplittingStrategy",
    "RandomSplit",
    "SubsamplePartitions",
    "AtomTypeSplit",
    "QCMLSplit",
]


def absolute_split_sizes(dsize: int, split_sizes: List[int]) -> List[int]:
    """
    Convert partition sizes to absolute values

    Args:
        dsize - Size of dataset.
        split_sizes - Sizes for each split. One can be set to -1 to assign all
            remaining data.
    """
    none_idx = None
    split_sizes = list(split_sizes)
    psum = 0

    for i in range(len(split_sizes)):
        s = split_sizes[i]
        if s is None or s < 0:
            if none_idx is None:
                none_idx = i
            else:
                raise ValueError(
                    f"Only one partition may be undefined (negative or None). "
                    f"Partition sizes: {split_sizes}"
                )
        else:
            if s < 1:
                split_sizes[i] = int(math.floor(s * dsize))

            psum += split_sizes[i]

    if none_idx is not None:
        remaining = dsize - psum
        split_sizes[none_idx] = int(remaining)

    return split_sizes


def random_split(dsize: int, *split_sizes: Union[int, float]) -> List[torch.tensor]:
    """
    Randomly split the dataset

    Args:
        dsize - Size of dataset.
        split_sizes - Sizes for each split. One can be set to -1 to assign all
            remaining data. Values in [0, 1] can be used to give relative partition
            sizes.
    """
    split_sizes = absolute_split_sizes(dsize, split_sizes)
    offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
    indices = torch.randperm(sum(split_sizes)).tolist()
    partition_sizes_idx = [
        indices[offset - length : offset]
        for offset, length in zip(offsets, split_sizes)
    ]
    return partition_sizes_idx


class SplittingStrategy:
    """
    Base class to implement various data splitting methods.
    """

    def __init__(self):
        pass

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        """
        Args:
            dataset - The dataset that is supposed to be split (an instance of BaseAtomsData).
            split_sizes - Sizes for each split. One can be set to -1 to assign all
                remaining data. Values in [0, 1] can be used to give relative partition
                sizes.

        Returns:
            list of partitions, where each one is a torch tensor with indices

        """
        raise NotImplementedError


class RandomSplit(SplittingStrategy):
    """
    Splitting strategy that partitions the data randomly into the given sizes
    """

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        dsize = len(dataset)
        partition_sizes_idx = random_split(dsize, *split_sizes)
        return partition_sizes_idx


class QCMLSplit(SplittingStrategy):
    """
    Splitting strategy for QCML dataset to have multiple conformers of one graph
    only in one set.
    Using percentage splits is recommanded.
    Args:
        external_metadata: path to the external metadata file
        drop_outliers: bool to drop outliers from the dataset
    """

    def __init__(self, external_metadata_path: str, drop_outliers: bool = False):
        self.external_metadata_path = external_metadata_path
        self.drop_outliers = drop_outliers

    def drop_is_outlier(self, idxs, outlier_idx):
        # function to drop outlier indices from the dataset if requested
        return idxs[~np.isin(idxs, outlier_idx)]

    def extract_indices(self, partition, group_ids):
        # function to extract db indices from smiles pointer
        # back to str to access dict keys
        idx = [str(n) for n in partition]
        final_idx = [group_ids[n] for n in idx]
        # concatenate all indices
        final_idx = np.concatenate(final_idx)
        return final_idx

    def split(self, dataset, *split_sizes):

        external_metadata = np.load(self.external_metadata_path, allow_pickle=True)
        # structure is key is pointer to smiles
        # the values are the db indices corresponding to the smiles
        group_ids = external_metadata["group_ids"].item()
        unique_smiles = list(group_ids.keys())
        # dict needed str as keys therefore back to int
        unique_smiles = [int(n) for n in unique_smiles]
        # number of dsize is number of unique smiles pointer
        dsize = len(unique_smiles)
        # partition the smiles pointer into requested sizes
        partition = random_split(len(unique_smiles), *split_sizes)
        # make indices for the database
        train_idx, val_idx, test_idx = [
            self.extract_indices(partition[i], group_ids) for i in range(3)
        ]
        print("Extracting indices done")

        if self.drop_outliers:
            print("Dropping outliers")
            is_outlier = sp.csr_matrix(
                (
                    external_metadata["is_outlier_data"],
                    external_metadata["is_outlier_indices"],
                    external_metadata["is_outlier_indptr"],
                ),
                shape=external_metadata["is_outlier_shape"],
            ).toarray()
            outlier_idx = np.where(is_outlier[0, :] == True)[0]
            clean_train_idx, clean_val_idx, clean_test_idx = [
                self.drop_is_outlier(idx, outlier_idx)
                for idx in (train_idx, val_idx, test_idx)
            ]
            partition_sizes_idx = [
                clean_train_idx.tolist(),
                clean_val_idx.tolist(),
                clean_test_idx.tolist(),
            ]

        else:
            partition_sizes_idx = [
                train_idx.tolist(),
                val_idx.tolist(),
                test_idx.tolist(),
            ]

        return partition_sizes_idx


class AtomTypeSplit(SplittingStrategy):
    """
    Strategy that filters out a specific atom type or multiple atom types from the database.
    And then performs the splitting on the filtered dataset.
    The remaining dataset are all molecules, except the ones that contain the atom type(s) to be filtered out.

    The data are read from the metadata.
    Data should be saved as sparse array, where the data,indices,pointer,shape are provided in metadata
    The keys in the metadata are of structure "atom_type_count_{indices OR indptr OR shape OR data}"
    Filter array is binary, where 1 means the atom type is present in the molecule and 0 means it is not.
    """

    def __init__(
        self,
        base_indices_path: str,
        draw_indices_path: str,
        external_metadata_path: str,
        drop_outliers: bool = True,
        num_draw: Union[int, float] = None,
    ):
        """
        Args:
            atomtypes: list of atom types to be filtered out.
            num_draw: percentage of the to be filtered out atomtypes to keep.
                        For now the percentage is applied to all atomtypes.
                        Values below 1 are interpreted as percentage, values above as absolute number.
                        Conversion is done automatically.
        """
        self.num_draw = num_draw
        self.drop_outliers = drop_outliers
        self.base_indices_path = base_indices_path
        self.draw_indices_path = draw_indices_path
        self.external_metadata_path = external_metadata_path

    def drop_is_outlier(self, idxs, outlier_idx):
        # function to drop outlier indices from the dataset if requested
        return idxs[~np.isin(idxs, outlier_idx)]

    def extract_indices(self, partition, group_ids):
        # function to extract db indices from smiles pointer
        # back to str to access dict keys
        idx = [str(n) for n in partition]
        final_idx = [group_ids[n] for n in idx]
        # concatenate all indices
        final_idx = np.concatenate(final_idx)
        return final_idx

    def split(self, dataset, *split_sizes):

        # external metadata storing atom type count, graph to conformer map and flat conformere to graph map
        external_metadata = np.load(self.external_metadata_path, allow_pickle=True)
        group_ids = external_metadata["group_ids"].item()
        conformere_to_graph = external_metadata["flat_conformere_to_graph_map.npy"]

        # load the indices of the database and draw indices
        base_indices = np.load(self.base_indices_path, allow_pickle=True)
        draw_indices = np.load(self.draw_indices_path, allow_pickle=True)

        # get unique graphs corresponding to base indices and create train test split from those
        unique_smiles = np.unique(conformere_to_graph[base_indices][:, 1])
        # number of dsize is number of unique smiles pointer
        dsize = len(unique_smiles)

        partition = random_split(len(unique_smiles), *split_sizes)
        # make indices for the database
        train_idx, val_idx, test_idx = [
            self.extract_indices(partition[i], group_ids) for i in range(3)
        ]
        print("Extracting indices done")

        if self.drop_outliers:
            print("Dropping outliers")
            is_outlier = sp.csr_matrix(
                (
                    external_metadata["is_outlier_data"],
                    external_metadata["is_outlier_indices"],
                    external_metadata["is_outlier_indptr"],
                ),
                shape=external_metadata["is_outlier_shape"],
            ).toarray()
            outlier_idx = np.where(is_outlier[0, :] == True)[0]
            clean_train_idx, clean_val_idx, clean_test_idx = [
                self.drop_is_outlier(idx, outlier_idx)
                for idx in (train_idx, val_idx, test_idx)
            ]
            partition_sizes_idx = [
                clean_train_idx.tolist(),
                clean_val_idx.tolist(),
                clean_test_idx.tolist(),
            ]

        else:
            partition_sizes_idx = [
                train_idx.tolist(),
                val_idx.tolist(),
                test_idx.tolist(),
            ]

        # get unique graphs corresponding to draw indices and place request num_draw to train idx, putting only one conformere per graph
        draw_unique_graphs = np.unique(conformere_to_graph[draw_indices][:, 1])
        # permutate them, they have no specific order from the start, but better to be sure
        np.random.shuffle(draw_unique_graphs)
        # take specified amount (either percentage or absolute number) of unique graphs
        if self.num_draw < 1:
            num_keep = int(math.floor(self.num_draw * draw_unique_graphs.shape[0]))
        else:
            num_keep = self.num_draw

        drawn = [group_ids[str(n)][0] for n in draw_unique_graphs[:num_keep]]

        partition_sizes_idx[0].extend(drawn)

        # # debugging only start
        # debug_indices = np.load("/home/elron/phd/projects/google/qmml/experiments/16/split_files/debug_indices.npy")
        # set_C = set(debug_indices[:,1].tolist())
        # set_A, set_B, set_D = [set(partition_sizes_idx[i]) for i in (0,1,2)]
        # set_A_new = set_A & set_C
        # set_B_new = set_B & set_C
        # set_D_new = set_D & set_C
        # result_dict = {row[1]: row[0] for row in debug_indices}
        # train_idx_new = [result_dict[n].item() for n in set_A_new]
        # val_idx_new = [result_dict[n].item() for n in set_B_new]
        # test_idx_new = [0]
        # partition_sizes_idx = [train_idx_new, val_idx_new, test_idx_new]
        # # debugging only end

        return partition_sizes_idx

    def random_split(
        self, indices, *split_sizes: Union[int, float]
    ) -> List[torch.tensor]:
        """
        Randomly split the dataset

        Args:
            dsize - Size of dataset.
            split_sizes - Sizes for each split. One can be set to -1 to assign all
                remaining data. Values in [0, 1] can be used to give relative partition
                sizes.
        """
        dsize = len(indices)
        split_sizes = absolute_split_sizes(dsize, split_sizes)
        offsets = torch.cumsum(torch.tensor(split_sizes), dim=0)
        indices = indices[torch.randperm(len(indices)).tolist()].tolist()
        partition_sizes_idx = [
            indices[offset - length : offset]
            for offset, length in zip(offsets, split_sizes)
        ]
        return partition_sizes_idx


class SubsamplePartitions(SplittingStrategy):
    """
    Strategy that splits the atoms dataset into predefined partitions as defined in the
    metadata. If the split size is smaller than the predefined partition, a given
    strategy will be used to subsample the partition (default: random).

    An metadata in the atoms dataset might look like this:

    >>> metadata = {
        my_partition_key : {
            "known": [0, 1, 2, 3],
            "test": [5, 6, 7]
        }
     }

    """

    def __init__(
        self,
        split_partition_sources: List[str],
        split_id=0,
        base_splitting: Optional[SplittingStrategy] = None,
        partition_key: str = "splits",
    ):
        """
        Args:
            split_partition_sources: names of partitions in metadata in the order of the
                supplied `split_sizes` in the `split` method. The same source can be
                used for multiple partitions. In that case the given `base_splitting`
                handles distribution the further splitting within each of the sources
                separately.
            split_id: the id of the predefined splitting
            base_splitting: method to subsample each partition
            partition_key: key in the metadata under which teh splitting is stores.
        """
        self.split_partition_sources = split_partition_sources
        self.partition_key = partition_key
        self.split_id = split_id

        self._unique_sources, self._splits_indices = np.unique(
            self.split_partition_sources, return_inverse=True
        )
        self.base_splitting = base_splitting or RandomSplit()

    def split(self, dataset, *split_sizes):
        if len(split_sizes) != len(self.split_partition_sources):
            raise ValueError(
                f"The number of `split_sizes`({len(split_sizes)}) needs to match the "
                + f"number of `partition_sources`({len(self.split_partition_sources)})."
            )

        split_partition_sizes = {src: [] for src in self.split_partition_sources}
        split_partition_idx = {src: [] for src in self.split_partition_sources}
        for i, split_size, src in zip(
            range(len(split_sizes)), split_sizes, self.split_partition_sources
        ):
            split_partition_sizes[src].append(split_size)
            split_partition_idx[src].append(i)

        partitions = dataset.metadata[self.partition_key]

        split_indices = [None] * len(split_sizes)
        for src in self._unique_sources:
            partition = partitions[src][self.split_id]
            print(len(partition))
            partition_split_indices = random_split(
                len(partition), *split_partition_sizes[src]
            )
            for i, split_idx in zip(split_partition_idx[src], partition_split_indices):
                split_indices[i] = np.array(partition)[split_idx].tolist()
        return split_indices


class GroupSplit(SplittingStrategy):
    """
    Strategy that splits the atoms dataset into non-overlapping groups, atoms under the same groups
    (setreoisomers/conformers) will be added to only one of the splits.

    the dictionary of groups is defined in the metadata under the key 'groups_ids' as follows:

    >>> metadata = {
        groups_ids : {
            "smiles_ids": [0, 1, 2, 3],
            "stereo_iso_id": [5, 6, 7],
            ...
        }
     }

    """

    def __init__(
        self,
        splitting_key: str,
        meta_key: str = "groups_ids",
        dataset_ids_key: Optional[str] = None,
    ):
        """
        Args:
            splitting_key: the id's key which will be used for the group splitting.
            meta_key: key in the metadata under which the groups ids and other ids are saved.
            dataset_ids_key: key in the metadata under which the ASE database ids are saved.
        """
        self.splitting_key = splitting_key
        self.meta_key = meta_key
        self.dataset_ids_key = dataset_ids_key

    def split(self, dataset, *split_sizes) -> List[torch.tensor]:
        md = dataset.metadata

        groups_ids = torch.tensor(md[self.meta_key][self.splitting_key])

        if len(dataset) != len(groups_ids) and dataset.subset_idx is None:
            raise ValueError(
                "The length of the dataset and the length of the groups ids are not equal."
            )

        # if the dataset is a subset of the original dataset, we need to map the groups ids to the subset ids
        if dataset.subset_idx is not None:
            _subset_ids = dataset.subset_idx
        else:
            _subset_ids = torch.arange(len(dataset))

        try:
            groups_ids = groups_ids[_subset_ids]
        except:
            raise ValueError(
                "the subset used of the dataset and the groups ids arrays provided doesn't match."
            )

        # check the split sizes
        unique_groups = torch.unique(groups_ids)
        dsize = len(unique_groups)
        sum_split_sizes = sum([s for s in split_sizes if s is not None and s > 0])

        if sum_split_sizes > dsize:
            raise ValueError(
                f"The sum of the splits sizes '{split_sizes}' should be less than "
                f"the number of available groups '{dsize}'."
            )

        # split the groups
        partitions = random_split(dsize, *split_sizes)
        partitions = [torch.isin(groups_ids, unique_groups[p]) for p in partitions]
        partitions = [(torch.where(p)[0]).tolist() for p in partitions]

        return partitions
