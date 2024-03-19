import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import timm  # 引入timm库以获取ViT模型
import os
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Cannot open {path}: {e}")
    else:
        return img

def setup_distributed():
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device  # 确保同时返回local_rank和device
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader):
        self.imgs = []
        self.labels = []
        self.transform = transform
        self.loader = loader
        with open(img_txt, 'r') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) == 41:
                    self.imgs.append(os.path.join(img_dir, parts[0]))
                    self.labels.append([int(value == '1') for value in parts[1:]])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return img, label

class FaceAttr(nn.Module):
    def __init__(self):
        super(FaceAttr, self).__init__()
        self.features = timm.create_model('vit_base_patch16_224', pretrained=True)
        # Assuming you only want the final feature vector, not the full sequence:
        self.features.head = nn.Identity()  # Remove the pre-trained head if it exists

        self.classifiers = nn.ModuleList([nn.Sequential(
            nn.Linear(768, 128),  # Adjust this to match the output size of your ViT model
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        ) for _ in range(40)])

    def forward(self, x):
        x = self.features(x)
        # Ensure x is of the correct shape (N, 768) before passing to classifiers
        return [classifier(x) for classifier in self.classifiers]


def train_and_validate(model, train_loader, val_loader, optimizer, checkpoint_path,
                       steps_per_checkpoint=1000, epochs=50):
    best_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = sum([nn.CrossEntropyLoss()(output, labels[:, i]) for i, output in enumerate(outputs)])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)
        if local_rank == 0:
            print(f"Epoch {epoch}, Loss: {average_loss}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                for i, output in enumerate(outputs):
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels[:, i]).sum().item()
            acc = correct / total
            val_accuracies.append(acc)
            if local_rank == 0:
                print(f'Validation Accuracy: {acc}')

            if acc > best_acc:
                best_acc = acc
                if local_rank == 0:
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'DDP_ViT_best_model_120epoch.pth'))
                    print('Best model saved')

        plot_metrics(train_losses, val_accuracies, checkpoint_path)


def test(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels[:, i]).sum().item()
    if local_rank == 0:
        print(f'Test Accuracy: {correct / total}')


def plot_metrics(train_losses, val_accuracies, save_path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Across Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Across Epochs')
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, 'ViT_celebA_training_validation_metrics.png'))
    print("metrics saved : ",save_path, 'ViT_celebA_training_validation_metrics.png')
    plt.close()


if __name__ == "__main__":
    # global device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank, device = setup_distributed()
    batch_size = 256


    train_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Annota_attGAN/train_label.txt'
    val_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Annota_attGAN/val_label.txt'
    test_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Annota_attGAN/test_label.txt'

    img_root = '/public/hezhenliang/users/gaoge/Data/CelebA/processed_img_celeba/celebA_align/data'

    checkpoint_path = ('/public/hezhenliang/users/gaoge/VIPLFaceMDT/VIPL_diffusion_hyperfeatures/process_celebA'
                       '/checkpoint_fullcelebA_ViT_DDP')

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = MyDataset(img_dir=img_root, img_txt=train_txt, transform=transform)
    # train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = MyDataset(img_dir=img_root, img_txt=val_txt, transform=transform)
    # val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 使用DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=val_sampler)


    test_dataset = MyDataset(img_dir=img_root, img_txt=test_txt, transform=transform)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = FaceAttr().to(device)
    model = DDP(model, device_ids=[device.index])

    # if torch.cuda.device_count() > 1:
    #     print(f"{torch.cuda.device_count()} GPUs are available.")
    #     model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    train_and_validate(model, train_loader, val_loader, optimizer, checkpoint_path, epochs=120)
    test(model, test_loader, os.path.join(checkpoint_path, 'DDP_ViT_best_model_120epoch.pth'))

