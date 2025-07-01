
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, Resize, CenterCrop
from torch.utils.data import DataLoader
import argparse
import os

def get_loaders(dataset, train_batch_size, test_batch_size, data_dir='./data/'):
    if dataset == 'mnist':
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x))])
        train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
        img_size = 28 * 28
    elif dataset == 'cifar10':
        transform = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Lambda(lambda x: torch.flatten(x))])
        train_dataset = CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
        img_size = 3 * 32 * 32
    elif dataset == 'cifar100':
        transform = Compose([
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            Lambda(lambda x: torch.flatten(x))])
        train_dataset = CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(data_dir, train=False, download=True, transform=transform)
        num_classes = 100
        img_size = 3 * 32 * 32
    elif dataset == 'imagenet':
        transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Lambda(lambda x: torch.flatten(x))
        ])
        imagenet_path = f'{data_dir}/imagenet'
        train_dataset = ImageFolder(f'{imagenet_path}/train', transform=transform)
        test_dataset = ImageFolder(f'{imagenet_path}/val', transform=transform)
        num_classes = 100 # Assuming ImageNet-100k means 100 classes
        img_size = 3 * 224 * 224
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader, num_classes, img_size


def overlay_y_on_x(x, y, num_classes):
    x_ = x.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class DS_FF_Layer(nn.Module):
    def __init__(self, in_features, out_features, num_classes, is_output_layer=False):
        super().__init__()
        # 主路径
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        
        # 辅助监督路径
        self.aux_classifier = nn.Sequential(
            nn.Linear(out_features, 128),  # 瓶颈层
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ) if not is_output_layer else None
        
        self.opt = Adam(self.parameters(), lr=0.01)
        self.threshold = nn.Parameter(torch.tensor(2.0))  # 可学习的阈值
        self.num_epochs = 500  # 减少每层训练轮次

    def forward(self, x):
        return self.main(x)

    def train(self, x_pos, x_neg, global_target=None):
        for _ in tqdm(range(self.num_epochs)):
            self.opt.zero_grad()

            h_pos = self.forward(x_pos)
            h_neg = self.forward(x_neg)

            g_pos = h_pos.pow(2).mean(1)
            g_neg = h_neg.pow(2).mean(1)
            ff_loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()

            if self.aux_classifier and global_target is not None:
                aux_output = self.aux_classifier(h_pos)
                ds_loss = F.kl_div(
                    F.log_softmax(aux_output, dim=1),
                    F.softmax(global_target, dim=1),
                    reduction='batchmean'
                )
                total_loss = ff_loss + 0.7 * ds_loss
            else:
                total_loss = ff_loss

            total_loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class DS_FF_Net(nn.Module):
    def __init__(self, dims, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        for d in range(len(dims) - 1):
            is_output_layer = (d == len(dims) - 2)
            self.layers.append(DS_FF_Layer(dims[d], dims[d + 1], num_classes, is_output_layer).cuda())

    def predict(self, x, batch_size=1000):
        goodness_per_label = []
        for label in range(self.num_classes):
            h = overlay_y_on_x(x, torch.tensor([label] * x.shape[0]).cuda(), self.num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        
        predictions = []
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i+batch_size]
            goodness_per_label = []
            for label in range(self.num_classes):
                h = overlay_y_on_x(x_batch, torch.tensor([label] * x_batch.shape[0]).cuda(), self.num_classes)
                goodness = []
                for layer in self.layers:
                    h = layer(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            predictions.append(goodness_per_label.argmax(1))
        return torch.cat(predictions)

    def generate_global_target(self, x, y):
        # A simple teacher model: one-hot encoding of the true labels
        return F.one_hot(y, self.num_classes).float()


def train_ds_ff_net(net, x_pos, x_neg, y):
    # 生成全局目标分布（教师模型）
    with torch.no_grad():
        global_target = net.generate_global_target(x_pos, y)
    
    h_pos, h_neg = x_pos, x_neg
    for i, layer in enumerate(net.layers):
        print(f'Training layer {i} with deep supervision...')
        # 仅在前80%层使用深度监督
        use_ds = i < len(net.layers) * 0.8
        h_pos, h_neg = layer.train(
            h_pos, h_neg,
            global_target if use_ds else None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Supervised Forward-Forward Algorithm Training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--train_batch_size', type=int, default=50000,
                        help='Training batch size (default: 50000)')
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help='Test batch size (default: 10000)')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory for storing data (default: ./data/)')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save the trained model weights to a file')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Load trained model weights from a file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training and only perform evaluation')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile for the model')
    args = parser.parse_args()

    torch.manual_seed(1234)
    train_loader, test_loader, num_classes, img_size = get_loaders(
        args.dataset, args.train_batch_size, args.test_batch_size, args.data_dir)

    net = DS_FF_Net([img_size, 500, 500, 500, 500], num_classes=num_classes)

    if not args.no_compile:
        print("Compiling the model...")
        net = torch.compile(net)

    if args.load_checkpoint:
        print(f"Loading model from {args.load_checkpoint}...")
        net.load_state_dict(torch.load(args.load_checkpoint))

    if not args.no_train:
        print("Starting training...")
        x, y = next(iter(train_loader))
        x, y = x.cuda(), y.cuda()

        x_pos = overlay_y_on_x(x, y, num_classes)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd], num_classes)

        train_ds_ff_net(net, x_pos, x_neg, y)
        print("Training complete.")

        if args.save_checkpoint:
            print("Saving model...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'ds_ff_{args.dataset}_net.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

        # Calculate and print training accuracy and error
        train_pred = net.predict(x)
        train_acc = train_pred.eq(y).float().mean().item()
        print(f'Train Accuracy: {train_acc:.4f}')
        print(f'Train Error: {1.0 - train_acc:.4f}')

    # Calculate and print test accuracy and error
    print("Evaluating on test set...")
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()
    
    test_pred = net.predict(x_te)
    test_acc = test_pred.eq(y_te).float().mean().item()
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Error: {1.0 - test_acc:.4f}')
