import os
import torch
from torch.utils.data import DataLoader
from classifier.data.datasets.kauto5cls import Kauto5Cls
from classifier.data.transform.transforms import AudioTransformer
from classifier.data.transform.target_transforms import TargetTransform



def build_dataset(
                cfg,
                data_path : str,
                transform = None,
                target_transform = None,
                is_train=True,
                is_inference=False
                ):
    """
    Function to generate dataset
    """
    dataset = None
    #Dataset directories are one level up and in directory named data
    assert os.path.exists(data_path),\
    f"directory: {data_path} not found in build_dataset"
    if cfg.INPUT.NAME == "kauto5cls":
        if not is_train:
            audio_bit_length = cfg.INPUT.RECORD_LENGTH/cfg.INPUT.SAMPLE_FREQ
        else:
            audio_bit_length = 0.0
        dataset = Kauto5Cls(
                    data_path,
                    transform,
                    target_transform,
                    is_train=is_train,
                    audio_bit_length=audio_bit_length
                    )
    return dataset
    pass





def make_data_loaders(cfg):
    """
    Makes dataloaders for training, validation and testing.
    Input:
        cfg - Config object, see source code for attributes
        max_iter - maximum number of iterations
        start_iter - start iteration number
    """
    #Code block to verify that data actually exists
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"
    dataset_names = ["train", "val", "test"]
    for dataset_name in dataset_names:
        if not os.path.exists(data_dir + cfg.INPUT.NAME + "/" + dataset_name):
            assert os.path.exists(data_dir + cfg.INPUT.NAME + "/download.sh"), \
            f"dataset: {cfg.INPUT.NAME} has no {dataset_name} data or download script"
            print(f"\n\n{dataset_name} dataset for {cfg.INPUT.NAME} not found, downloading\n\n")
            os.system(data_dir + cfg.INPUT.NAME + "/download.sh")
        assert os.path.exists(data_dir + cfg.INPUT.NAME + "/" + dataset_name), \
        f"download script tried, verify that it downloads train, validate, and test subdirs"
    
    #Create transform objects
    train_transform = AudioTransformer(cfg.INPUT.TRANSFORM, is_train=True)
    val_test_transform = AudioTransformer(cfg.INPUT.TRANSFORM, is_train = False)
    target_transform = TargetTransform(cfg)
    train_dataset = build_dataset(
                        cfg,
                        data_path=data_dir + cfg.INPUT.NAME + "/train",
                        transform=train_transform,
                        target_transform=target_transform
                        )
    val_dataset = build_dataset(
                        cfg,
                        data_path=data_dir + cfg.INPUT.NAME + "/val",
                        transform=val_test_transform,
                        target_transform=target_transform,
                        is_train=False
                        )
    test_dataset = build_dataset(
                        cfg,
                        data_path=data_dir + cfg.INPUT.NAME + "/test",
                        transform=val_test_transform,
                        target_transform=target_transform,
                        is_train=False
                        )
    
    #Set up samplers and batch samplers
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    train_batch_size = cfg.TRAINER.BATCH_SIZE
    val_batch_size = cfg.TEST.BATCH_SIZE
    test_batch_size = cfg.TEST.BATCH_SIZE
    train_batch_sampler = torch.utils.data.sampler.BatchSampler(
                            sampler=train_sampler,
                            batch_size=train_batch_size,
                            drop_last=True
                            )
    val_batch_sampler = torch.utils.data.sampler.BatchSampler(
                            sampler=val_sampler,
                            batch_size=val_batch_size,
                            drop_last=False
                            )
    test_batch_sampler = torch.utils.data.sampler.BatchSampler(
                            sampler=test_sampler,
                            batch_size=test_batch_size,
                            drop_last=False
                            )
    #Instantiate dataloaders and return
    train_loader = DataLoader(
                    train_dataset,
                    num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                    batch_sampler=train_batch_sampler,
                    pin_memory=cfg.DATA_LOADER.PIN_MEMORY
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                    batch_sampler=val_batch_sampler,
                    pin_memory=cfg.DATA_LOADER.PIN_MEMORY
                    )
    test_loader = DataLoader(
                    test_dataset,
                    num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                    batch_sampler=test_batch_sampler,
                    pin_memory=cfg.DATA_LOADER.PIN_MEMORY
                    )
    return train_loader, val_loader, test_loader
