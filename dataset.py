import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = os.path.join(root_dir, 'LR')
        self.hr_dir = os.path.join(root_dir, 'HR')
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        else:
            lr_img = transforms.ToTensor()(lr_img) # Apply ToTensor if no transform is provided

        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)
        else:
            hr_img = transforms.ToTensor()(hr_img) # Apply ToTensor if no transform is provided

        return lr_img, hr_img

if __name__ == '__main__':
    # Example usage:
    root_dir = 'dataset/train'  # Replace with your actual dataset root directory
    transform_lr_example = transforms.Compose([
        transforms.RandomCrop(128),
        transforms.ToTensor()
    ])
    transform_hr_example = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    dataset_example = PairedImageDataset(root_dir, transform_lr=transform_lr_example, transform_hr=transform_hr_example)
    dataloader_example = torch.utils.data.DataLoader(dataset_example, batch_size=4, shuffle=True)

    for i, (lr_batch, hr_batch) in enumerate(dataloader_example):
        print(f"Batch {i}:")
        print("LR batch shape:", lr_batch.shape)
        print("HR batch shape:", hr_batch.shape)
        if i == 2:  # Just print a few batches for example
            break