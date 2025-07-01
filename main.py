import torch
import torch.nn as nn
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


class Net(torch.nn.Module):

    def __init__(self, dims, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        for d in range(len(dims) - 1):
            self.layers.append(Layer(dims[d], dims[d + 1]).cuda())

    def predict(self, x):
        goodness_per_label = []
        for label in range(self.num_classes):
            h = overlay_y_on_x(x, label, self.num_classes)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forward-Forward Algorithm Training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use (default: mnist)')
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

    net = Net([img_size, 500, 500], num_classes=num_classes)
    
    if not args.no_compile:
        import time
        print("Compiling the model...")
        start_time = time.time()
        net = torch.compile(net)
        end_time = time.time()
        print(f"Model compiled in {end_time - start_time:.2f}s")

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

        net.train(x_pos, x_neg)
        print("Training complete.")

        if args.save_checkpoint:
            print("Saving model...")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'ff_{args.dataset}_net.pth')
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
