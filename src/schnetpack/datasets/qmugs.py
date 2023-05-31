import io
import logging
import os
import re
import shutil
import tarfile
import tempfile
from typing import List, Optional, Dict, TextIO
from urllib import request as request
import random
import pickle
import calendar
import time


import numpy as np
from ase import Atoms
import ase
from ase.io.extxyz import read_xyz
#from ase.io.sdf import read_sdf
from tqdm import tqdm


import torch
from schnetpack import properties
from schnetpack.data import (
    AtomsDataModule,
    AtomsDataFormat,
    ASEAtomsData,
    BaseAtomsData,
    create_dataset,
    load_dataset,
    AtomsDataModuleError,
    SplittingStrategy,
)
from schnetpack.data.qmugs_split_strategy import QMugsSplit
__all__ = ["QMugs",]




class QMugs(AtomsDataModule):
    """
    QMugs benchmark database for drug-like molecules.
    Properties are on GFN2 and/or DFT level (def2-SVP / wB97X-D) of theory 
    Naming convention is level of theory prefix and property at the end
    The QMugs database contains small organic molecules with up to 100 non-hydrogen atoms
    from including C, O, N, F, P, S, Cl, Br, I. 

    This class adds convenient functions to download QMugs from ETH Zürich Research collection and load the data into pytorch.
    The wavefunctions are not downloaded
    Additionally it ensures that all conformers of a molecule are assigned to the same set (train, val or test) to avoid data leakage

    This is currently work in progress
    References:

        .. [#qmugs] https://www.research-collection.ethz.ch/handle/20.500.11850/482129

    """



    # GFN2 vars
    chemblid = 'CHEMBL_ID'
    conf_id = 'CONF_ID'

    tot_energy_GFN2 ='GFN2:TOTAL_ENERGY'
    atomic_energy_GFN2 ='GFN2:ATOMIC_ENERGY'
    formation_energy_GFN2 ='GFN2:FORMATION_ENERGY'
    tot_H_GFN2 ='GFN2:TOTAL_ENTHALPY'
    tot_G_GFN2 ='GFN2:TOTAL_FREE_ENERGY'


    tot_mu_GFN2 ='GFN2:DIPOLE_MOMENT'
    vec_mu_GFN2 = "GFN2:DIPOLE_VECTOR_COMPONENTS"
    Qij_GFN2 ='GFN2:QUADRUPOLE'
 

    A_GFN2 = "GFN2:ROT_CONSTANT_A"
    B_GFN2 = "GFN2:ROT_CONSTANT_B"
    C_GFN2 = "GFN2:ROT_CONSTANT_C"


    vib_enthalphy_GFN2 = 'GFN2:ENTHALPY_VIBRATIONAL'
    rot_enthalphy_GFN2 = 'GFN2:ENTHALPY_ROTATIONAL'
    transl_enthalphy_GFN2 = 'GFN2:ENTHALPY_TRANSLATIONAL'
    tot_enthalphy_GFN2 = 'GFN2:ENTHALPY_TOTAL'


    vib_Cv_GFN2 = 'GFN2:HEAT_CAPACITY_VIBRATIONAL'
    rot_Cv_GFN2 = 'GFN2:HEAT_CAPACITY_ROTATIONAL'
    transl_Cv_GFN2 = 'GFN2:HEAT_CAPACITY_TRANSLATIONAL'
    tot_Cv_GFN2 = 'GFN2:HEAT_CAPACITY_TOTAL'

    vib_S_GFN2 = 'GFN2:ENTROPY_VIBRATIONAL'
    rot_S_GFN2 = 'GFN2:ENTROPY_ROTATIONAL'
    transl_S_GFN2 = 'GFN2:ENTROPY_TRANSLATIONAL'
    tot_S_GFN2 = 'GFN2:ENTROPY_TOTAL'

         
    homo_GFN2 = 'GFN2:HOMO_ENERGY'
    lumo_GFN2 ='GFN2:LUMO_ENERGY'
    gap_GFN2 ='GFN2:HOMO_LUMO_GAP'
    fermlvl_GFN2 ='GFN2:FERMI_LEVEL'
    mulliken_GFN2 ='GFN2:MULLIKEN_CHARGES'
    cov_coord_num_GFN2 ='GFN2:COVALENT_COORDINATION_NUMBER'
    disp_coeff_mol_GFN2 ='GFN2:DISPERSION_COEFFICIENT_MOLECULAR'
    disp_coeff_atom_GFN2 ='GFN2:DISPERSION_COEFFICIENT_ATOMIC'
    alpha_mol ='GFN2:POLARIZABILITY_MOLECULAR'
    alpha_atom ='GFN2:POLARIZABILITY_ATOMIC'
    Mab_GFN2 ='GFN2:WIBERG_BOND_ORDER'
    tot_Mab_GFN2 ='GFN2:TOTAL_WIBERG_BOND_ORDER'

    # DFT vars
    tot_energy_DFT ='DFT:TOTAL_ENERGY'
    atomic_energy_DFT ='DFT:ATOMIC_ENERGY'
    formation_energy_DFT ='DFT:FORMATION_ENERGY'
    vesp_DFT ='DFT:ESP_AT_NUCLEI'
    cl_GFN2 ='DFT:LOWDIN_CHARGES'
    cm_GFN2 ='DFT:MULLIKEN_CHARGES'

    A_DFT = "DFT:ROT_CONSTANT_A"
    B_DFT = "DFT:ROT_CONSTANT_B"
    C_DFT = "DFT:ROT_CONSTANT_C"      
    tot_mu_DFT = "DFT:DIPOLE_MOMENT"
    vec_mu_DFT = "DFT:DIPOLE_VECTOR_COMPONENTS"

    Exc_DFT ='DFT:XC_ENERGY'
    vnuc_DFT ='DFT:NUCLEAR_REPULSION_ENERGY'
    te_DFT ='DFT:ONE_ELECTRON_ENERGY'
    vee_DFT ='DFT:TWO_ELECTRON_ENERGY'
    vnn_DFT = 'DFT:NUCLEAR_NUCLEAR_REPULSION_ENERGY' # not in dataset calculated internally
    homo_DFT ='DFT:HOMO_ENERGY'
    lumo_DFT ='DFT:LUMO_ENERGY'
    gap_DFT ='DFT:HOMO_LUMO_GAP'
    mab_DFT ='DFT:MAYER_BOND_ORDER'
    wab_DFT ='DFT:WIBERG_LOWDIN_BOND_ORDER'
    tot_mab_DFT ='DFT:TOTAL_MAYER_BOND_ORDER'
    tot_wab_DFT ='DFT:TOTAL_WIBERG_LOWDIN_BOND_ORDER'

    
    # names of the properties saved in the mol sdf file
    QMugs_properties = [
                    'CHEMBL_ID',
                    'CONF_ID',
                    'GFN2:TOTAL_ENERGY',
                    'GFN2:ATOMIC_ENERGY',
                    'GFN2:FORMATION_ENERGY',
                    'GFN2:TOTAL_ENTHALPY',
                    'GFN2:TOTAL_FREE_ENERGY',
                    'GFN2:DIPOLE',
                    'GFN2:QUADRUPOLE',
                    'GFN2:ROT_CONSTANTS',
                    'GFN2:ENTHALPY',
                    'GFN2:HEAT_CAPACITY',
                    'GFN2:ENTROPY',
                    'GFN2:HOMO_ENERGY',
                    'GFN2:LUMO_ENERGY',
                    'GFN2:HOMO_LUMO_GAP',
                    'GFN2:FERMI_LEVEL',
                    'GFN2:MULLIKEN_CHARGES',
                    'GFN2:COVALENT_COORDINATION_NUMBER',
                    'GFN2:DISPERSION_COEFFICIENT_MOLECULAR',
                    'GFN2:DISPERSION_COEFFICIENT_ATOMIC',
                    'GFN2:POLARIZABILITY_MOLECULAR',
                    'GFN2:POLARIZABILITY_ATOMIC',
                    'GFN2:WIBERG_BOND_ORDER',
                    'GFN2:TOTAL_WIBERG_BOND_ORDER',
                    'DFT:TOTAL_ENERGY',
                    'DFT:ATOMIC_ENERGY',
                    'DFT:FORMATION_ENERGY',
                    'DFT:ESP_AT_NUCLEI',
                    'DFT:LOWDIN_CHARGES',
                    'DFT:MULLIKEN_CHARGES',
                    'DFT:ROT_CONSTANTS',
                    'DFT:DIPOLE',
                    'DFT:XC_ENERGY',
                    'DFT:NUCLEAR_REPULSION_ENERGY',
                    'DFT:ONE_ELECTRON_ENERGY',
                    'DFT:TWO_ELECTRON_ENERGY',
                    'DFT:NUCLEAR_NUCLEAR_REPULSION_ENERGY'
                    'DFT:HOMO_ENERGY',
                    'DFT:LUMO_ENERGY',
                    'DFT:HOMO_LUMO_GAP',
                    'DFT:MAYER_BOND_ORDER',
                    'DFT:WIBERG_LOWDIN_BOND_ORDER',
                    'DFT:TOTAL_MAYER_BOND_ORDER',
                    'DFT:TOTAL_WIBERG_LOWDIN_BOND_ORDER',
                    properties.Z,
                    properties.R,
                    properties.cell,
                    properties.pbc
                ]

    # dict for which QMugs properties a splitting into single seperate components will be made
    QMugs_exception_list_map = {
            'GFN2:DIPOLE':['GFN2:DIPOLE_VECTOR_COMPONENTS','GFN2:DIPOLE_MOMENT'],
            'DFT:DIPOLE':['DFT:DIPOLE_VECTOR_COMPONENTS','DFT:DIPOLE_MOMENT'],
            'GFN2:ROT_CONSTANTS':["GFN2:ROT_CONSTANT_A","GFN2:ROT_CONSTANT_B","GFN2:ROT_CONSTANT_C"],
            'DFT:ROT_CONSTANTS':["DFT:ROT_CONSTANT_A","DFT:ROT_CONSTANT_B","DFT:ROT_CONSTANT_C"],
            'GFN2:ENTHALPY':[
                            'GFN2:ENTHALPY_VIBRATIONAL',
                            'GFN2:ENTHALPY_ROTATIONAL',
                            'GFN2:ENTHALPY_TRANSLATIONAL',
                            'GFN2:ENTHALPY_TOTAL'],
            'GFN2:HEAT_CAPACITY' : [
                            'GFN2:HEAT_CAPACITY_VIBRATIONAL',
                            'GFN2:HEAT_CAPACITY_ROTATIONAL',
                            'GFN2:HEAT_CAPACITY_TRANSLATIONAL',
                            'GFN2:HEAT_CAPACITY_TOTAL'],
            'GFN2:ENTROPY': [
                            'GFN2:ENTROPY_VIBRATIONAL',
                            'GFN2:ENTROPY_ROTATIONAL',
                            'GFN2:ENTROPY_TRANSLATIONAL',
                            'GFN2:ENTROPY_TOTAL'],

            } #             'GFN2:QUADRUPOLE': ['GFN:QUADRUPOLE_VECTOR_COMPONENTS','GFN2:QUADRUPOLE_MOMENT'],


    def __init__(
        self,
        datapath: str,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 4,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        predefined_props: Optional[str] = None,
        atomrefs: Dict[str, torch.Tensor] = None,
        splitting: Optional[QMugsSplit] = None,
        **kwargs   
    ):

        """
        Args:
            datapath: path to dataset
            batch_size: (train) batch size
            num_train: number of training examples
            num_val: number of validation examples
            num_test: number of test examples
            split_file: path to npz file with data partitions
            format: dataset format
            load_properties: subset of properties to load
            remove_uncharacterized: do not include uncharacterized molecules.
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
            predefined_props: set of properties that should be written to database
        """
        super().__init__(
            datapath=datapath,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            splitting=splitting,
            **kwargs
        )

        self.atomrefs = atomrefs
        if predefined_props:
            self.predefined_props = predefined_props + [properties.Z, properties.R, properties.cell, properties.pbc]
        else:
            self.predefined_props = None

    # TODO include atomrefs for now it is not included atomrefs
    def prepare_data(self):
        if not os.path.exists(self.datapath):
            property_unit_dict = {
                
                # atomic units and none units are set to 1
                # GFN2 vars units        
                QMugs.chemblid : None,
                QMugs.conf_id : None,
                QMugs.tot_energy_GFN2 : "Ha",
                QMugs.atomic_energy_GFN2 : "Ha",
                QMugs.formation_energy_GFN2 : "Ha",
                QMugs.tot_H_GFN2 : "Ha",
                QMugs.tot_G_GFN2 : "Ha",
                QMugs.vec_mu_GFN2 : "D",
                QMugs.tot_mu_GFN2 : "D",

                QMugs.Qij_GFN2 : "D*A",
                QMugs.A_GFN2 : "1/cm",
                QMugs.B_GFN2 : "1/cm",
                QMugs.C_GFN2 : "1/cm",
                QMugs.vib_enthalphy_GFN2 : "cal/mol",
                QMugs.rot_enthalphy_GFN2 : "cal/mol",
                QMugs.transl_enthalphy_GFN2 : "cal/mol",
                QMugs.vib_S_GFN2 : "cal/mol/K",
                QMugs.rot_S_GFN2 : "cal/mol/K",
                QMugs.transl_S_GFN2 : "cal/mol/K",
                QMugs.tot_S_GFN2 : "cal/mol/K",
                QMugs.tot_enthalphy_GFN2 : "cal/mol",
                QMugs.vib_Cv_GFN2 : "cal/mol/K",
                QMugs.rot_Cv_GFN2 : "cal/mol/K",
                QMugs.transl_Cv_GFN2 : "cal/mol/K",
                QMugs.tot_Cv_GFN2 : "cal/mol/K",
                QMugs.homo_GFN2 : "Ha",
                QMugs.lumo_GFN2 : "Ha",
                QMugs.gap_GFN2 : "Ha",
                QMugs.fermlvl_GFN2 : "Ha",
                QMugs.mulliken_GFN2 : 1,
                QMugs.cov_coord_num_GFN2 : 1,
                QMugs.disp_coeff_mol_GFN2 : 1,
                QMugs.disp_coeff_atom_GFN2 : 1,
                QMugs.alpha_mol : 1,
                QMugs.alpha_atom : 1,
                QMugs.Mab_GFN2 : 1,
                QMugs.tot_Mab_GFN2 : 1,
                # DFT vars units
                QMugs.tot_energy_DFT : "Ha",
                QMugs.atomic_energy_DFT : "Ha",
                QMugs.formation_energy_DFT : "Ha",
                QMugs.vesp_DFT : "kgm^2/s^3A",
                QMugs.cl_GFN2 : 1,
                QMugs.cm_GFN2 : 1,
                QMugs.A_DFT : "1/cm",
                QMugs.B_DFT : "1/cm",
                QMugs.C_DFT : "1/cm",
                QMugs.tot_mu_DFT : "D",
                QMugs.vec_mu_DFT : "D",
                QMugs.Exc_DFT : "Ha",
                QMugs.vnuc_DFT : "Ha",
                QMugs.te_DFT : "Ha",
                QMugs.vee_DFT : "Ha",
                QMugs.vnn_DFT: "Ha",
                QMugs.homo_DFT : "Ha",
                QMugs.lumo_DFT : "Ha",
                QMugs.gap_DFT : "Ha",
                QMugs.mab_DFT : 1,
                QMugs.wab_DFT : 1,
                QMugs.tot_mab_DFT : 1,
                QMugs.tot_wab_DFT : 1
            }
            
            if self.predefined_props is not None:
                predefined_props_flag = True

                self.QMugs_properties = list(filter(lambda p: p in self.predefined_props, list(property_unit_dict.keys())))
                for p in list(filter(lambda p: p not in self.predefined_props, list(property_unit_dict.keys()))):
                    if not p in [QMugs.chemblid,QMugs.conf_id]:
                        property_unit_dict.pop(p)

            else:
                predefined_props_flag = False
            tmpdir = tempfile.mkdtemp("qmugs")
            logging.info(f"tempdir created at {tmpdir}")


            dataset = create_dataset(
                datapath=self.datapath,
                format=self.format,
                distance_unit="Ang",
                property_unit_dict=property_unit_dict,
                atomrefs=self.atomrefs)
        
            self._download_data(tmpdir, dataset,predefined_props_flag)
            self.dataset_idx_to_conformere_map()
            shutil.rmtree(tmpdir)

        else:
            dataset = load_dataset(self.datapath, self.format)
            
            if "QMugs" in list(dataset.metadata.keys()):
                pass
            else:
                self.dataset_idx_to_conformere_map()
                print("Done")
            self._copy_to_workdir()

    def dataset_idx_to_conformere_map(self):

        logging.info(f"Adding db id to chembl_id map to metadata key ""QMugs""")
        with ASEAtomsData(self.datapath).conn as db:
            
            # first col is database idx and second is numeric CHEMBL_ID
            data = {}

            # chemblID to str conversion needed otherwise json object not serializable
            val = list({db.get(i).id -1: str(db.get(i).data["CHEMBL_ID"][0]) for i in tqdm(range(1,len(db)+1))}.items())
            data["QMugs"] = val
            md = db.metadata
            md.update(data)
            db.metadata = md
        logging.info(f"Done with updating metadata")

    def _download_atomrefs(self, tmpdir):
        NotImplementedError

    def get_num_atoms_sdf_v2000(self,first_line: str) -> int:
        """Parse the first line extracting the number of atoms."""
        return int(first_line[0:3])  # first three characters
        # http://biotech.fyicenter.com/1000024_SDF_File_Format_Specification.html

    def read_sdf(self,sdf_path) -> Atoms:

        with open(sdf_path,"r") as file_obj:
            """Read the sdf data and compose the corresponding Atoms object."""
            lines = file_obj.readlines()
            # first three lines header
            del lines[:3]

            num_atoms = self.get_num_atoms_sdf_v2000(lines.pop(0))
            positions = []
            symbols = []
            for line in lines[:num_atoms]:
                x, y, z, symbol = line.split()[:4]
                symbols.append(symbol)
                positions.append((float(x), float(y), float(z)))
            return Atoms(symbols=symbols, positions=positions)

    def split_into_single_componts(self,mol,p):

        prop = np.array([float(x) for x in mol.GetProp(p).split("|")])

        # .reshape(1,3) is for dipole moment to have correct dimension
        if len(self.QMugs_exception_list_map[p]) == 2:
            val = [prop[:-1].reshape(1,3),np.array(prop[-1])]
        else:
            val = [np.array(prop[n]) for n in range(len(prop))]

        return val

    def _flatten_list(self,folder):

        return[item for sublist in folder for item in sublist]

    def _download_data(
        self, tmpdir, dataset: BaseAtomsData,
    predefined_props_flag = Optional[bool]):
        # TODO uncomment after testing, we just dont want to download the data all the time
        try:
            from rdkit import Chem

        except:
            raise ImportError(
                "In order to download QMugs data, you have to install "
                "rdkit"
            )
        # logging.info(f"Downloading QMugs data (excluding wavefunction files) to temporary directory {tmpdir}...")
        # tar_path = os.path.join(tmpdir, "structures.tar.gz")
        # raw_path = os.path.join(tmpdir, "structures")
        # url = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz"
        # request.urlretrieve(url, tar_path)
        # logging.info("Done.")

        # logging.info("Extracting files...")
        # tar = tarfile.open(tar_path)
        # tar.extractall(tmpdir)
        # tar.close()
        # logging.info("Done.")

        # add convert sdf to ASE atoms object -> internal ASE function does not work correctly --> TODO check if works correct now
        # create file with database idx associated with same molecule different conformers
        ##
        # TODO remove raw_path after succesful testing
        # TODO but for later replace rdkit sdf functionality buy custom to circumvent additional package
        raw_path  = "/home/elron/phd/benchmark_data/QMugs/structures/"
        logging.info("Parse sdf files and all properties...")


        ### TMP ADDED 
        # 1 get folder names
        N = 100000
        folder = self._flatten_list([ 
            [os.path.join(raw_path,f,r) for r in os.listdir(os.path.join(raw_path,f))]
            for f in tqdm(os.listdir(raw_path))
            ]) # !TODO remove []:10000] after test was successful

        to_exclude = ["no_path_dummy_vars"]
        try:
            with open("/home/elron/phd/benchmark_data/QMugs/already_in_db.pickle","rb") as handle:
                if handle:
                    to_exclude = pickle.load(handle)
                    to_exclude = []
                else:
                    to_exclude = []
                    print("Blub")
        except:
            pass
            to_exclude = []

        folder.sort(reverse=True)
        C = [n for n in folder if n not in to_exclude ]
        samples = random.sample(C,N)

        with open("/home/elron/phd/benchmark_data/QMugs/already_in_db.pickle","wb") as handle:
            pickle.dump(to_exclude.extend(samples),handle)

        chunk_size = [int((len(samples) / 10)*j) for j in range(0,10)]
        chunk_size.extend([-1])

        for i in range(len(chunk_size)-1):
 
            self._load_sdf_chunk(samples[chunk_size[i]:chunk_size[i+1]],dataset,predefined_props_flag)


    def _load_sdf_chunk(self,chunk,dataset,predefined_props_flag):
            
            try:
                from rdkit import Chem

            except:
                raise ImportError(
                    "In order to create a QMugs database, you have to install "
                    "rdkit"
                )



            CONF_DICT = {
            "conf_00":np.array([0]),
            "conf_01":np.array([1]),
            "conf_02":np.array([2])
            }

            property_list = []
            
            if predefined_props_flag:
                n = None
                m = None
            else:
                n = 2
                m = -4

            for sdf_path in tqdm(chunk):

                # 2 make props dict, reading molecule from sdf file

                props = dict.fromkeys([x for x in self.QMugs_properties if x not in list(self.QMugs_exception_list_map.keys())])
                

                mol = next(Chem.SDMolSupplier(sdf_path,removeHs=False))
                mol.GetConformer()
                atms = self.read_sdf(sdf_path)

                props[properties.Z] = atms.numbers
                props[properties.R] = atms.positions
                props[properties.cell] = atms.cell
                props[properties.pbc] = atms.pbc

                for p in [n for n in self.QMugs_properties[n:m] if not "NUCLEAR_NUCLEAR" in n]:

                    if p in self.QMugs_exception_list_map:
                        val = self.split_into_single_componts(mol,p)

                        for i, new_p in enumerate(self.QMugs_exception_list_map[p]):
                            props[new_p] = val[i]


                    elif (type(mol.GetProp(p)) == str):
                        props[p] = np.array([float(x) for x in mol.GetProp(p).split("|")])

                    else: 
                        props[p] = np.array(mol.GetProp(p))

                # Find a better way to include is only temporally for me 
                if "DFT:NUCLEAR_NUCLEAR_REPULSION_ENERGY" in self.QMugs_properties:
                    props["DFT:NUCLEAR_NUCLEAR_REPULSION_ENERGY"] = (
                        props["DFT:TOTAL_ENERGY"] - 
                        props["DFT:XC_ENERGY"] - 
                        props["DFT:NUCLEAR_REPULSION_ENERGY"] -
                        props["DFT:ONE_ELECTRON_ENERGY"] -
                        props["DFT:TWO_ELECTRON_ENERGY"]
                    )
                else:
                    props["DFT:NUCLEAR_NUCLEAR_REPULSION_ENERGY"] = np.array([0.])
                # splitting because iterating through numerical tensors is easier. all numbers are unique IDs
                # makes later grouping of conformers easier
                props["CHEMBL_ID"] = np.array([int(mol.GetProp("CHEMBL_ID").split("CHEMBL")[-1])])
                props["CONF_ID"] = CONF_DICT[mol.GetProp("CONF_ID")]
                property_list.append(props)

            logging.info(f"Write atoms to db at {self.datapath}...")
            dataset.add_systems(property_list=property_list)
            logging.info("Done.")


"""
Notes: The check_connectivity functions from the datamodule is a bottleneck in combination with
a large database (~1GB) and the num of workers set high
The bigger the database is the slower the single threads are in checking the connectivity
On my machine the whole QMugs Database (all properties + structures ~ 25GB) is extremly slow 
in checking the connectivity. Additionally if the num workers is set higher than 4 the RAM is overflown and 
my machine is shuting down.

--> loading only a subindex of the dataset (via load_dataset(with provided subindex) does not circumvent the issue
--> for now I circumvent this by creating multiple Databases
--> setup a ensemble kind of way training style
--> even if I can load the whole db into RAM on the cluster, the checking for connectivity will be slow, since parallel functions
are not implemented in the ASE.db class
--> This is especially bad in the beginning when I have to determine the best placement cutoff 
--> Plus when testing if the origin token should be turned off
--> Plus when I want to initializae conditon only on one condition

"""
################


