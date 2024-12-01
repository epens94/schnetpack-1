import logging
import os
import uuid
import tempfile
import socket
from typing import List
import random

import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.logger import Logger

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.data import BaseAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter
from schnetpack import properties
from schnetpack.utils import load_model
import subprocess
from subprocess import check_output


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()),use_cache=True)
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = """
   _____      __    _   __     __  ____             __
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    if "run_gcs" in config.globals.keys():

        #########
        ## all database downloading stuff

        # for my custom restarting from checkpoint from gs cloud, only use the ID
        DB_FOLDER,BUCKET_NAME = "/home/data/databases","qcml_transfer_learning"

        # since container will always new initialized
        os.makedirs(DB_FOLDER, exist_ok=True)
        # download the database
        db_name = os.path.basename(config.data.datapath)
        db_folder = os.path.join(DB_FOLDER, db_name)
        # for google cloud necessarcy ?
        subprocess.check_call(['gcloud','storage','cp','-r',
                                config.globals.db_storage_path,
                                db_folder])
        print("download database done")
        config.data.datapath = db_folder

        ##########
        ## all checkpoint downloading stuff

        if config.run.ckpt_path is not None:
            CKP_FOLDER = os.path.join(config.run.path,config.run.id,"checkpoints")
            logging.info(f"{CKP_FOLDER}")
            logging.info(f"{config.run.id}")
            config.run.id = config.run.ckpt_path
            logging.info(f"{config.run.id}")
            # now we download the checkpoint from the cloud
            command = f'gcloud storage ls gs://{BUCKET_NAME}/experiments/{config.run.ckpt_path}/checkpoints/'
            ckpt_list = [n for n in check_output(command, shell=True, text=True).strip().split("\n") if "epoch" in n]

            # which ckptoint to choose
            tag1,tag2 = ("and_of_epoch","periodic_")

            cond = {
                tag: [int(f.split("epoch=")[-1].split(".")[0]),f]
                for f in ckpt_list
                for tag in (tag1, tag2)
                if tag in f
            }
            # either periodic ckpt or end of epoch ckpt
            ckpt_folder = cond[tag2][1] if cond[tag2][0] >= cond[tag1][0] else cond[tag1][1]

            # download the checkpoint
            resume_ckpt_path = os.path.join(CKP_FOLDER, os.path.basename(ckpt_folder))
            resume_split_path = os.path.join(CKP_FOLDER,"split.npz")
            subprocess.check_call(['gcloud','storage','cp','-r',
                                    ckpt_folder,
                                    resume_ckpt_path])
            
            # download the split file
            command = f'gs://{BUCKET_NAME}/experiments/{config.run.ckpt_path}/split.npz'
            subprocess.check_call(['gcloud','storage','cp', '-r',
                                    command,
                                    resume_split_path])
            # update the config
            config.data.split_file = resume_split_path
            config.run.ckpt_path = resume_ckpt_path
            log.info(
                    f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
                )

    log.info("Running on host: " + str(socket.gethostname()))
    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        # choose seed randomly
        with open_dict(config):
            config.seed = random.randint(0, 2**32 - 1)
        log.info(f"Seed randomly with <{config.seed}>")
    seed_everything(seed=config.seed, workers=True)

    if config.get("print_config"):
        print_config(config, resolve=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.data,
        train_sampler_cls=(
            str2class(config.data.train_sampler_cls)
            if config.data.train_sampler_cls
            else None
        ),
    )

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)
                if "Wandb" in lg_conf._target_:
                    l.config = OmegaConf.to_container(config, resolve=True)
                    sorted_config = dict(sorted(OmegaConf.to_container(config, resolve=True).items()))
                else:
                    sorted_config = config
                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=sorted_config, model=task, trainer=trainer)

    # Train the model
    log.info("Starting training.")

    # Handling resuming checkpoint when in the middle of training epoch stopped
    # since the atomsmodule is not resumable

    if config.run.ckpt_path is not None:

        # manually overwrite the batch progress to ensure that all remaining indices are used
        checkpoint = torch.load(config.run.ckpt_path,map_location="cpu")
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"] = 0
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["ready"] = 0

        torch.save(checkpoint,config.run.ckpt_path)
        # finish the last running epoch
        trainer.fit(model=task, datamodule=datamodule,ckpt_path=config.run.ckpt_path)

        log.info("Re-Instantiating datamodule after finishing last ran epoch")
        BASE = os.listdir(os.path.join(config.run.path,config.run.id,"checkpoints"))
        ckpt_path = [os.path.join(BASE,f) for f in BASE if "ckpt_at_and_of" in f][0]

        # Re Init everything, necessary because EMA makes problems and it is cleaner
        # Init Lightning datamodule
        log.info(f"RE-Instantiating datamodule <{config.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(
            config.data,
            train_sampler_cls=(
                str2class(config.data.train_sampler_cls)
                if config.data.train_sampler_cls
                else None
            )
        )

        # Init model
        log.info(f"RE-Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        # Init LightningModule
        log.info(f"RE-Instantiating task <{config.task._target_}>")
        scheduler_cls = (
            str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
        )

        task: spk.AtomisticTask = hydra.utils.instantiate(
            config.task,
            model=model,
            optimizer_cls=str2class(config.task.optimizer_cls),
            scheduler_cls=scheduler_cls,
        )

        # Init Lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"RE-Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init Lightning loggers
        logger: List[Logger] = []

        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if "_target_" in lg_conf:
                    log.info(f"RE-Instantiating logger <{lg_conf._target_}>")
                    l = hydra.utils.instantiate(lg_conf)
                    if "Wandb" in lg_conf._target_:
                        l.config = OmegaConf.to_container(config, resolve=True)
                        sorted_config = dict(sorted(OmegaConf.to_container(config, resolve=True).items()))
                    else:
                        sorted_config = config
                    logger.append(l)

        # Init Lightning trainer
        log.info(f"RE-Instantiating trainer <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            default_root_dir=os.path.join(config.run.id),
            _convert_="partial",
            #reload_dataloaders_every_n_epochs=1
        )
        log_hyperparameters(config=sorted_config, model=task, trainer=trainer)
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=ckpt_path)
        print("Done")

    else:
        trainer.fit(model=task, datamodule=datamodule, ckpt_path=None)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")

    # for google cloud services and using wandb
    if "wandb" in config.logger.keys():
        import wandb
        wandb.finish()

