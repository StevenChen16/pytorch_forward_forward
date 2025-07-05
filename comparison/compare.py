import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt

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

class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train_epoch(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), loss.item()

class FFNet(torch.nn.Module):
    def __init__(self, dims, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        for d in range(len(dims) - 1):
            self.layers.append(FFLayer(dims[d], dims[d + 1]).cuda())

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

    def forward_with_activations(self, x):
        activations = []
        h = x
        for layer in self.layers:
            h = layer(h)
            activations.append(h.detach().clone())
        return h, activations

    def train_epoch(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        avg_loss = 0.0
        for i, layer in enumerate(self.layers):
            h_pos, h_neg, loss = layer.train_epoch(h_pos, h_neg)
            avg_loss += loss
        return avg_loss / len(self.layers)

    def get_layer_info(self):
        layer_info = []
        for i, layer in enumerate(self.layers):
            layer_info.append({
                'layer_idx': i,
                'weight': layer.weight.detach().clone(),
                'bias': layer.bias.detach().clone(),
            })
        return layer_info

class BPNet(torch.nn.Module):
    def __init__(self, dims, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_classes = num_classes
        
        for d in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[d], dims[d + 1]).cuda())
        
        self.output_layer = nn.Linear(dims[-1], num_classes).cuda()
        
        self.relu = nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
            h = self.relu(h)
        h = self.output_layer(h)
        return h

    def forward_with_activations(self, x):
        activations = []
        h = x
        
        for layer in self.layers:
            h = layer(h)
            h = self.relu(h)
            activations.append(h.detach().clone())
        
        h = self.output_layer(h)
        activations.append(h.detach().clone())
        
        return h, activations

    def train_epoch(self, x_combined, y_combined):
        self.opt.zero_grad()
        output = self.forward(x_combined)
        loss = self.criterion(output, y_combined)
        loss.backward()
        self.opt.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return output.argmax(1)

    def get_layer_info(self):
        layer_info = []
        for i, layer in enumerate(self.layers):
            layer_info.append({
                'layer_idx': i,
                'weight': layer.weight.detach().clone(),
                'bias': layer.bias.detach().clone()
            })
        layer_info.append({
            'layer_idx': len(self.layers),
            'weight': self.output_layer.weight.detach().clone(),
            'bias': self.output_layer.bias.detach().clone()
        })
        return layer_info

class NetworkComparator:
    def __init__(self, save_dir='./comparison_results_v3'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def compare_weights(self, ff_layers, bp_layers, epoch=None):
        weight_comparisons = []
        min_layers = min(len(ff_layers), len(bp_layers))
        for i in range(min_layers):
            ff_weight = ff_layers[i]['weight']
            bp_weight = bp_layers[i]['weight']
            weight_diff = torch.abs(ff_weight - bp_weight)
            weight_comparison = {
                'layer_idx': i, 'epoch': epoch,
                'weight_diff_mean': weight_diff.mean().item(),
                'weight_diff_std': weight_diff.std().item(),
                'ff_weight_norm': ff_weight.norm().item(),
                'bp_weight_norm': bp_weight.norm().item(),
                'weight_cosine_sim': F.cosine_similarity(ff_weight.flatten(), bp_weight.flatten(), dim=0).item()
            }
            weight_comparisons.append(weight_comparison)
        return weight_comparisons

    def compare_activations(self, ff_activations, bp_activations, epoch=None):
        activation_comparisons = []
        min_layers = min(len(ff_activations), len(bp_activations))
        for i in range(min_layers):
            ff_act, bp_act = ff_activations[i], bp_activations[i]
            if ff_act.shape != bp_act.shape:
                min_size = min(ff_act.shape[1], bp_act.shape[1])
                ff_act, bp_act = ff_act[:, :min_size], bp_act[:, :min_size]
            act_diff = torch.abs(ff_act - bp_act)
            activation_comparison = {
                'layer_idx': i, 'epoch': epoch,
                'activation_diff_mean': act_diff.mean().item(),
                'activation_correlation': torch.corrcoef(torch.stack([ff_act.flatten(), bp_act.flatten()]))[0, 1].item()
            }
            activation_comparisons.append(activation_comparison)
        return activation_comparisons

    def compare_performance(self, ff_net, bp_net, test_loader, epoch=None):
        ff_correct, bp_correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                ff_pred, bp_pred = ff_net.predict(x), bp_net.predict(x)
                ff_correct += (ff_pred == y).sum().item()
                bp_correct += (bp_pred == y).sum().item()
                total += y.size(0)
        return {
            'epoch': epoch,
            'ff_accuracy': ff_correct / total,
            'bp_accuracy': bp_correct / total,
            'total_samples': total
        }

    def save_comparison_results(self, all_results, filename='comparison_results.json'):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Comparison results saved to {filepath}")

    def plot_comparisons(self, results):
        epochs = [res['epoch'] for res in results['performance_comparisons']]
        
        plt.figure(figsize=(18, 10))
        
        # Accuracy
        plt.subplot(2, 3, 1)
        plt.plot(epochs, [res['ff_accuracy'] for res in results['performance_comparisons']], label='Forward-Forward')
        plt.plot(epochs, [res['bp_accuracy'] for res in results['performance_comparisons']], label='Backpropagation')
        plt.title('Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(2, 3, 2)
        plt.plot(epochs, results['ff_losses'], label='FF Loss')
        plt.plot(epochs, results['bp_losses'], label='BP Loss')
        plt.title('Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        num_layers = len(results['weight_comparisons'][0])
        
        # Weight Diff
        plt.subplot(2, 3, 3)
        for i in range(num_layers):
            plt.plot(epochs, [res[i]['weight_diff_mean'] for res in results['weight_comparisons']], label=f'Layer {i}')
        plt.title('Weight Difference (Mean)')
        plt.xlabel('Epoch')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True)

        # Weight Cosine Similarity
        plt.subplot(2, 3, 4)
        for i in range(num_layers):
            plt.plot(epochs, [res[i]['weight_cosine_sim'] for res in results['weight_comparisons']], label=f'Layer {i}')
        plt.title('Weight Cosine Similarity')
        plt.xlabel('Epoch')
        plt.ylabel('Similarity')
        plt.legend()
        plt.grid(True)

        # Activation Diff
        plt.subplot(2, 3, 5)
        for i in range(num_layers):
            plt.plot(epochs, [res[i]['activation_diff_mean'] for res in results['activation_comparisons']], label=f'Layer {i}')
        plt.title('Activation Difference (Mean)')
        plt.xlabel('Epoch')
        plt.ylabel('Difference')
        plt.legend()
        plt.grid(True)
        
        # Activation Correlation
        plt.subplot(2, 3, 6)
        for i in range(num_layers):
            plt.plot(epochs, [res[i]['activation_correlation'] for res in results['activation_comparisons']], label=f'Layer {i}')
        plt.title('Activation Correlation')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'comparison_plots.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Forward-Forward vs Backpropagation Comparison')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'], help='Dataset to use')
    parser.add_argument('--train_batch_size', type=int, default=5000, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=10000, help='Test batch size')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Directory for storing data')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--comparison-dir', type=str, default='./comparison_results_v3', help='Directory to save comparison results')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print("Loading data...")
    train_loader, test_loader, num_classes, img_size = get_loaders(
        args.dataset, args.train_batch_size, args.test_batch_size, args.data_dir)
    
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    x_pos = overlay_y_on_x(x, y, num_classes)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd], num_classes)
    
    comparator = NetworkComparator(save_dir=args.comparison_dir)
    all_results = {
        'ff_losses': [], 'bp_losses': [], 'weight_comparisons': [],
        'activation_comparisons': [], 'performance_comparisons': [],
        'training_info': {
            'dataset': args.dataset, 'num_epochs': args.num_epochs,
            'batch_size': args.train_batch_size, 'img_size': img_size,
            'num_classes': num_classes
        }
    }

    print("Initializing models...")
    dims = [img_size, 500, 500]
    ff_net = FFNet(dims, num_classes=num_classes)
    bp_net = BPNet(dims, num_classes=num_classes)

    print("Compiling models...")
    ff_net = torch.compile(ff_net)
    bp_net = torch.compile(bp_net)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        ff_loss = ff_net.train_epoch(x_pos, x_neg)
        bp_loss = bp_net.train_epoch(x, y)
        
        all_results['ff_losses'].append(ff_loss)
        all_results['bp_losses'].append(bp_loss)

        ff_layer_info = ff_net.get_layer_info()
        bp_layer_info = bp_net.get_layer_info()

        all_results['weight_comparisons'].append(
            comparator.compare_weights(ff_layer_info, bp_layer_info, epoch=epoch)
        )
        
        with torch.no_grad():
            _, ff_activations = ff_net.forward_with_activations(x_pos)
            _, bp_activations = bp_net.forward_with_activations(x)
        
        all_results['activation_comparisons'].append(
            comparator.compare_activations(ff_activations, bp_activations, epoch=epoch)
        )

        performance_comp = comparator.compare_performance(ff_net, bp_net, test_loader, epoch=epoch)
        all_results['performance_comparisons'].append(performance_comp)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.num_epochs} | Time: {epoch_time:.2f}s | "
              f"FF Loss: {ff_loss:.4f} | BP Loss: {bp_loss:.4f} | "
              f"FF Acc: {performance_comp['ff_accuracy']:.4f} | BP Acc: {performance_comp['bp_accuracy']:.4f}")

        if (epoch + 1) % 100 == 0:
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            ff_path = os.path.join(args.comparison_dir, f'ff_net_epoch_{epoch+1}.pth')
            bp_path = os.path.join(args.comparison_dir, f'bp_net_epoch_{epoch+1}.pth')
            torch.save(ff_net.state_dict(), ff_path)
            torch.save(bp_net.state_dict(), bp_path)

    print("Training complete.")
    print("Saving final models...")
    torch.save(ff_net.state_dict(), os.path.join(args.comparison_dir, 'ff_net_final.pth'))
    torch.save(bp_net.state_dict(), os.path.join(args.comparison_dir, 'bp_net_final.pth'))

    comparator.save_comparison_results(all_results)
    print("Plotting results...")
    try:
        comparator.plot_comparisons(all_results)
    except Exception as e:
        print(f"Error plotting comparisons: {e}")

    print("\nComparison completed!")

if __name__ == "__main__":
    main()
