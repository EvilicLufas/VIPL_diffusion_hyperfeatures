import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

import extract_hyperfeatures
from extract_hyperfeatures import *  # 假设函数名为extract_features

# Import necessary libraries
import torch
from PIL import Image
from omegaconf import OmegaConf

# Assuming other necessary imports from your project structure are here
from extract_hyperfeatures import load_models
from archs.correspondence_utils import process_image

# Set the device for computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load configuration and models
config_path = "configs/real.yaml"  # Update the path as necessary
config, diffusion_extractor, aggregation_network = load_models(config_path, device)


def extract_img_hyperfeats(image_path, device="cuda", load_size=(178, 218)):
    # Load and process the image
    img_pil = Image.open(image_path).convert("RGB")
    img, _ = process_image(img_pil, res=load_size)
    img = img.to(device)

    # Extract and aggregate features
    with torch.inference_mode():
        with torch.autocast(device):
            feats, _ = diffusion_extractor.forward(img[None, ...])  # Add a batch dimension
            b, s, l, w, h = feats.shape
            hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))

    return hyperfeats


# 定义转换、数据集和模型
def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Cannot open {path}: {e}")
    else:
        return img


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_txt, transform=None, loader=default_loader, ):
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

        # 使用特征提取函数
        img_feature = extract_img_hyperfeats(image_path=img, device="cuda",
                                             load_size=(178, 218))

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return img_feature, label


def make_conv():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def make_fc():
    return nn.Sequential(
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 2)
    )


class FaceAttr(nn.Module):
    def __init__(self):
        super(FaceAttr, self).__init__()
        # Set the input size as a constant, example: 512
        self.features = nn.Linear(512, 128)  # Adjust 512 to your specific extracted feature size
        self.classifiers = nn.ModuleList([self.make_fc() for _ in range(40)])

    def make_fc(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return [classifier(x) for classifier in self.classifiers]


# class FaceAttr(nn.Module):
#     def __init__(self, input_size):  # 假设提取的特征有不同的大小
#         super(FaceAttr, self).__init__()
#         self.features = nn.Sequential(
#             nn.Linear(input_size, 128),  # 调整为与提取特征匹配的尺寸
#             nn.ReLU(),
#             # 之后的层保持不变...
#         )
#         self.classifiers = nn.ModuleList([make_fc() for _ in range(40)])
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return [classifier(x) for classifier in self.classifiers]

# 训练和验证模型
def train_and_validate(model, train_loader, val_loader, optimizer, save_path, steps_per_checkpoint=1000, epochs=50):
    best_acc = 0
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

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                for i, output in enumerate(outputs):
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels[:, i]).sum().item()
            acc = correct / total
            print(f'Validation Accuracy: {acc}')

            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print('Best model saved')

            # Save accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(epoch, acc, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(save_path, 'with_hyper/accuracy_plot.png'))
            plt.close()


# 测试模型
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
    print(f'Test Accuracy: {correct / total}')


# main函数
def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义变量

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file",
                        default="configs/real.yaml")  # "configs/real.yaml"
    parser.add_argument("--images_or_prompts_path", type=str,
                        help="Path to json with image paths (real) or prompts (synthetic)",
                        default="annotations/spair_71k_test-6.json")  # "annotations/synthetic-3.json"
    parser.add_argument("--save_root", type=str, help="Root directory to save the results", default="hyperfeatures")
    parser.add_argument("--image_root", type=str, help="Root directory storing images (real)",
                        default="assets/spair/images")  # ""
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)

    train_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Anno/10000_list_attr_celeba_train.txt'
    val_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Anno/10000_list_attr_celeba_val.txt'
    test_txt = '/public/hezhenliang/users/gaoge/Data/CelebA/Anno/10000_list_attr_celeba_test.txt'

    img_root = '/public/hezhenliang/users/gaoge/Data/CelebA/Img/img_align_celeba'

    checkpoint_path = '/public/hezhenliang/users/gaoge//VIPLFaceMDT/VIPL_diffusion_hyperfeatures/process_celebA/checkpoint'

    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 初始化数据集和数据加载器
    train_dataset = MyDataset(img_dir=img_root, img_txt=train_txt, transform=transform)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = MyDataset(img_dir=img_root, img_txt=val_txt, transform=transform)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dataset = MyDataset(img_dir=img_root, img_txt=test_txt, transform=transform)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 检测是否有多个GPU可用
    model = FaceAttr().to(device)
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs are available.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    # 调用训练、验证和测试的函数
    train_and_validate(model, train_loader, val_loader, optimizer, checkpoint_path, epochs=50)
    test(model, test_loader, os.path.join(checkpoint_path, 'with_hyper/best_model.pth'))


# # Example usage
# if __name__ == "__main__":
#     image_path = "path/to/your/image.jpg"  # Update this path to your image
#     hyperfeats = extract_img_hyperfeats(image_path, device, load_size=(178, 218))
#     print(hyperfeats.shape)


if __name__ == "__main__":
    main()
