import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import config


def get_class_names(train_dir: str = config.TRAIN_DIR) -> list:
    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Training folder not found at {train_dir}. "
            "Set BRAIN_MRI_DATA_ROOT to the correct dataset location."
        )
    return sorted(os.listdir(train_dir))  #this function right here, actually returns a random order of the image classes


def get_transforms(train: bool) -> transforms.Compose:
    pipeline = [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE))]
    if train:
        #simple data augmentation
        pipeline += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
    
            
    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    return transforms.Compose(pipeline)


def build_datasets(class_names: list):
    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=get_transforms(train=True))
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=get_transforms(train=False))

    if set(train_dataset.classes) != set(class_names):
        raise ValueError("Class order mismatch -- ImageFolder order must match class_names.")

    return train_dataset, test_dataset
    
#And ImageFolder labels by sorted order.
#So get_class_names and ImageFolder image class order disagree. So a sorted function will fix it.


def build_dataloaders(train_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, test_loader


def check_train_test_overlap(train_dir: str = config.TRAIN_DIR, test_dir: str = config.TEST_DIR) -> set:
    def all_filenames(root):
        names = set()
        for cls in os.listdir(root):
            class_dir = os.path.join(root, cls)
            if os.path.isdir(class_dir):
                names.update(os.listdir(class_dir))
        return names

    overlap = all_filenames(train_dir) & all_filenames(test_dir)
    if overlap:
        print(f"[WARNING] {len(overlap)} filenames appear in both Training/ and Testing/.")
    else:
        print("[OK] No exact filename overlap between Training/ and Testing/.")
    return overlap