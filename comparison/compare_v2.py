import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import psutil
import GPUtil
import threading
from torchprofile import profile_macs
import torchvision.models as models

# 从compare.py导入组件
from compare import BPNet, FFNet, FFLayer, overlay_y_on_x, get_loaders

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PowerMonitor:
    """功耗监控类"""
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """监控资源使用"""
        while self.monitoring:
            # CPU和内存使用率
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            
            # GPU使用率
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_usage.append(gpu.load * 100)
                    self.gpu_memory.append(gpu.memoryUsed)
            except:
                pass
            
            time.sleep(0.1)
    
    def get_average_usage(self):
        """获取平均使用率"""
        return {
            'cpu_avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'memory_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'gpu_avg': np.mean(self.gpu_usage) if self.gpu_usage else 0,
            'gpu_memory_avg': np.mean(self.gpu_memory) if self.gpu_memory else 0
        }

def get_loaders_v2(dataset, train_batch_size, test_batch_size, data_dir='./data/'):
    """获取数据加载器 - 增强版"""
    if dataset == 'mnist':
        transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
        img_shape = (1, 28, 28)
    elif dataset == 'cifar10':
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = Compose([
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        num_classes = 10
        img_shape = (3, 32, 32)
    elif dataset == 'cifar100':
        transform_train = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = Compose([
            ToTensor(),
            Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dataset = CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = CIFAR100(data_dir, train=False, download=True, transform=transform_test)
        num_classes = 100
        img_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader, num_classes, img_shape

def overlay_y_on_x_v2(x, y, num_classes):
    """FF算法的标签嵌入 - 支持多维输入"""
    if len(x.shape) > 2:
        # 对于图像数据，先展平
        x_flat = x.view(x.size(0), -1)
    else:
        x_flat = x
    
    x_ = x_flat.clone()
    x_[:, :num_classes] *= 0.0
    x_[range(x.shape[0]), y] = x_flat.max()
    
    # 如果原始输入是多维的，保持原始形状
    if len(x.shape) > 2:
        return x_.view(x.size())
    else:
        return x_

# ============== 全连接网络 (使用原版) ==============
class FFNet_V2(FFNet):
    """继承原版FFNet，增加一些方法"""
    def __init__(self, dims, num_classes=10):
        super().__init__(dims, num_classes)
    
    def forward_with_activations(self, x):
        """前向传播并记录激活值"""
        activations = []
        # 确保输入是扁平的
        if len(x.shape) > 2:
            h = x.view(x.size(0), -1)
        else:
            h = x
        
        for layer in self.layers:
            h = layer(h)
            activations.append(h.detach().clone())
        return h, activations

class BPNet_V2(BPNet):
    """继承原版BPNet，增加一些方法"""
    def __init__(self, dims, num_classes=10):
        super().__init__(dims, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 确保输入是扁平的
        if len(x.shape) > 2:
            h = x.view(x.size(0), -1)
        else:
            h = x
        
        for layer in self.layers:
            h = layer(h)
            h = self.relu(h)
        h = self.output_layer(h)
        return h

    def forward_with_activations(self, x):
        """前向传播并记录激活值"""
        activations = []
        # 确保输入是扁平的
        if len(x.shape) > 2:
            h = x.view(x.size(0), -1)
        else:
            h = x
        
        for layer in self.layers:
            h = layer(h)
            h = self.relu(h)
            activations.append(h.detach().clone())
        
        h = self.output_layer(h)
        activations.append(h.detach().clone())
        
        return h, activations

    def train_epoch(self, x_combined, y_combined):
        """训练一个epoch"""
        self.opt.zero_grad()
        output = self.forward(x_combined)
        loss = self.criterion(output, y_combined)
        loss.backward()
        self.opt.step()
        return loss.item()

# ============== LeNet-5 ==============
class LeNet5_FF(nn.Module):
    """LeNet-5的FF版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 卷积层（保持标准BP训练）
        self.conv1 = nn.Conv2d(3, 6, 5).cuda()
        self.conv2 = nn.Conv2d(6, 16, 5).cuda()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # FF全连接层
        self.fc_layers = nn.ModuleList([
            FFLayer(16 * 5 * 5, 120).cuda(),
            FFLayer(120, 84).cuda()
        ])
        
        # 卷积层的优化器
        self.conv_opt = Adam(list(self.conv1.parameters()) + list(self.conv2.parameters()), lr=0.001)

    def conv_forward(self, x):
        """卷积部分前向传播"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x.view(-1, 16 * 5 * 5)

    def predict(self, x):
        conv_out = self.conv_forward(x)
        goodness_per_label = []
        for label in range(self.num_classes):
            h = overlay_y_on_x_v2(conv_out, torch.full((conv_out.size(0),), label, device=x.device), self.num_classes)
            goodness = []
            for layer in self.fc_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def forward_with_activations(self, x):
        """前向传播并记录激活值"""
        activations = []
        
        # 卷积层
        conv_out = self.conv_forward(x)
        activations.append(conv_out.detach().clone())
        
        # FF层
        h = conv_out
        for layer in self.fc_layers:
            h = layer(h)
            activations.append(h.detach().clone())
        
        return h, activations

    def train_epoch(self, x_pos, x_neg, y_pos, y_neg):
        # 训练卷积层
        conv_out_pos = self.conv_forward(x_pos)
        conv_out_neg = self.conv_forward(x_neg)
        
        # 为卷积层创建监督信号（简化版）
        conv_out_combined = torch.cat([conv_out_pos, conv_out_neg])
        y_combined = torch.cat([y_pos, y_neg])
        
        # 简化的卷积层损失
        conv_output = self.relu(conv_out_combined)  # 简单的激活
        conv_loss = F.mse_loss(conv_output.mean(dim=1), y_combined.float())
        
        self.conv_opt.zero_grad()
        conv_loss.backward()
        self.conv_opt.step()
        
        # 训练FF层
        h_pos = overlay_y_on_x_v2(conv_out_pos.detach(), y_pos, self.num_classes)
        h_neg = overlay_y_on_x_v2(conv_out_neg.detach(), y_neg, self.num_classes)
        
        avg_loss = conv_loss.item()
        for layer in self.fc_layers:
            h_pos, h_neg, loss = layer.train_epoch(h_pos, h_neg)
            avg_loss += loss
        return avg_loss / (len(self.fc_layers) + 1)

    def get_layer_info(self):
        layer_info = []
        # 卷积层信息
        layer_info.append({'layer_idx': 0, 'weight': self.conv1.weight.detach().clone(), 
                          'bias': self.conv1.bias.detach().clone()})
        layer_info.append({'layer_idx': 1, 'weight': self.conv2.weight.detach().clone(), 
                          'bias': self.conv2.bias.detach().clone()})
        # FF层信息
        for i, layer in enumerate(self.fc_layers):
            layer_info.append({'layer_idx': i+2, 'weight': layer.weight.detach().clone(), 
                              'bias': layer.bias.detach().clone()})
        return layer_info

class LeNet5_BP(nn.Module):
    """LeNet-5的BP版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5).cuda()
        self.conv2 = nn.Conv2d(6, 16, 5).cuda()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120).cuda()
        self.fc2 = nn.Linear(120, 84).cuda()
        self.fc3 = nn.Linear(84, num_classes).cuda()
        
        self.opt = Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_activations(self, x):
        activations = []
        x = self.pool(self.relu(self.conv1(x)))
        activations.append(x.detach().clone())
        x = self.pool(self.relu(self.conv2(x)))
        activations.append(x.detach().clone())
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        activations.append(x.detach().clone())
        x = self.relu(self.fc2(x))
        activations.append(x.detach().clone())
        x = self.fc3(x)
        activations.append(x.detach().clone())
        return x, activations

    def train_epoch(self, x_combined, y_combined):
        self.opt.zero_grad()
        output = self.forward(x_combined)
        loss = self.criterion(output, y_combined)
        loss.backward()
        self.opt.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(1)

    def get_layer_info(self):
        return [
            {'layer_idx': 0, 'weight': self.conv1.weight.detach().clone(), 'bias': self.conv1.bias.detach().clone()},
            {'layer_idx': 1, 'weight': self.conv2.weight.detach().clone(), 'bias': self.conv2.bias.detach().clone()},
            {'layer_idx': 2, 'weight': self.fc1.weight.detach().clone(), 'bias': self.fc1.bias.detach().clone()},
            {'layer_idx': 3, 'weight': self.fc2.weight.detach().clone(), 'bias': self.fc2.bias.detach().clone()},
            {'layer_idx': 4, 'weight': self.fc3.weight.detach().clone(), 'bias': self.fc3.bias.detach().clone()}
        ]

# ============== ResNet-101 ==============
class ResNet101_FF(nn.Module):
    """ResNet-101的FF版本（简化版）"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        
        # 使用预训练的ResNet-101作为特征提取器
        self.feature_extractor = models.resnet101(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # 移除最后的全连接层
        
        # 冻结部分层
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 只训练最后几层
        for param in self.feature_extractor.layer4.parameters():
            param.requires_grad = True
        
        # FF全连接层
        self.fc_layers = nn.ModuleList([
            FFLayer(2048, 1024).cuda(),
            FFLayer(1024, 512).cuda()
        ])
        
        # 特征提取器的优化器
        self.feature_opt = Adam(filter(lambda p: p.requires_grad, self.feature_extractor.parameters()), lr=0.0001)
        
        self.feature_extractor = self.feature_extractor.cuda()

    def extract_features(self, x):
        """提取特征"""
        return self.feature_extractor(x)

    def predict(self, x):
        features = self.extract_features(x)
        goodness_per_label = []
        for label in range(self.num_classes):
            h = overlay_y_on_x_v2(features, torch.full((features.size(0),), label, device=x.device), self.num_classes)
            goodness = []
            for layer in self.fc_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def forward_with_activations(self, x):
        """前向传播并记录激活值"""
        features = self.extract_features(x)
        activations = [features.detach().clone()]
        
        h = features
        for layer in self.fc_layers:
            h = layer(h)
            activations.append(h.detach().clone())
        
        return h, activations

    def train_epoch(self, x_pos, x_neg, y_pos, y_neg):
        # 提取特征
        features_pos = self.extract_features(x_pos)
        features_neg = self.extract_features(x_neg)
        
        # 训练特征提取器
        features_combined = torch.cat([features_pos, features_neg])
        y_combined = torch.cat([y_pos, y_neg])
        
        # 简化的特征损失
        feature_loss = F.mse_loss(features_combined.mean(dim=1), y_combined.float())
        self.feature_opt.zero_grad()
        feature_loss.backward()
        self.feature_opt.step()
        
        # 训练FF层
        h_pos = overlay_y_on_x_v2(features_pos.detach(), y_pos, self.num_classes)
        h_neg = overlay_y_on_x_v2(features_neg.detach(), y_neg, self.num_classes)
        
        avg_loss = feature_loss.item()
        for layer in self.fc_layers:
            h_pos, h_neg, loss = layer.train_epoch(h_pos, h_neg)
            avg_loss += loss
        return avg_loss / (len(self.fc_layers) + 1)

    def get_layer_info(self):
        return [{'layer_idx': i, 'weight': layer.weight.detach().clone(), 
                'bias': layer.bias.detach().clone()} for i, layer in enumerate(self.fc_layers)]

class ResNet101_BP(nn.Module):
    """ResNet-101的BP版本"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Linear(2048, num_classes)
        
        # 冻结部分层
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 只训练最后几层
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
        self.resnet = self.resnet.cuda()
        self.opt = Adam(filter(lambda p: p.requires_grad, self.resnet.parameters()), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.resnet(x)

    def forward_with_activations(self, x):
        # 简化的激活值提取
        features = self.resnet(x)
        return features, [features.detach().clone()]

    def train_epoch(self, x_combined, y_combined):
        self.opt.zero_grad()
        output = self.forward(x_combined)
        loss = self.criterion(output, y_combined)
        loss.backward()
        self.opt.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).argmax(1)

    def get_layer_info(self):
        return [{'layer_idx': 0, 'weight': self.resnet.fc.weight.detach().clone(), 
                'bias': self.resnet.fc.bias.detach().clone()}]

class AdvancedNetworkComparator:
    """高级网络比较器"""
    def __init__(self, save_dir='./comparison_results_v2'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.power_monitor = PowerMonitor()

    def compute_flops(self, model, input_shape):
        """计算模型的FLOPs"""
        try:
            dummy_input = torch.randn(1, *input_shape).cuda()
            macs = profile_macs(model, dummy_input)
            return macs * 2  # MAC to FLOPs
        except Exception as e:
            print(f"Error computing FLOPs: {e}")
            return 0

    def compare_weights(self, ff_layers, bp_layers, epoch=None):
        """比较权重"""
        weight_comparisons = []
        min_layers = min(len(ff_layers), len(bp_layers))
        for i in range(min_layers):
            ff_weight = ff_layers[i]['weight']
            bp_weight = bp_layers[i]['weight']
            
            # 处理不同形状的权重
            if ff_weight.shape != bp_weight.shape:
                continue
            
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
        """比较激活值"""
        activation_comparisons = []
        min_layers = min(len(ff_activations), len(bp_activations))
        for i in range(min_layers):
            ff_act = ff_activations[i].view(ff_activations[i].size(0), -1)
            bp_act = bp_activations[i].view(bp_activations[i].size(0), -1)
            
            if ff_act.shape != bp_act.shape:
                min_size = min(ff_act.shape[1], bp_act.shape[1])
                ff_act = ff_act[:, :min_size]
                bp_act = bp_act[:, :min_size]
            
            try:
                act_diff = torch.abs(ff_act - bp_act)
                correlation = torch.corrcoef(torch.stack([ff_act.flatten(), bp_act.flatten()]))[0, 1].item()
                activation_comparison = {
                    'layer_idx': i, 'epoch': epoch,
                    'activation_diff_mean': act_diff.mean().item(),
                    'activation_correlation': correlation if not torch.isnan(torch.tensor(correlation)) else 0.0
                }
                activation_comparisons.append(activation_comparison)
            except:
                continue
        return activation_comparisons

    def compare_performance(self, ff_net, bp_net, test_loader, epoch=None):
        """比较性能"""
        ff_correct, bp_correct, total = 0, 0, 0
        ff_predictions, bp_predictions, true_labels = [], [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                ff_pred = ff_net.predict(x)
                bp_pred = bp_net.predict(x)
                
                ff_correct += (ff_pred == y).sum().item()
                bp_correct += (bp_pred == y).sum().item()
                total += y.size(0)
                
                ff_predictions.extend(ff_pred.cpu().numpy())
                bp_predictions.extend(bp_pred.cpu().numpy())
                true_labels.extend(y.cpu().numpy())
        
        return {
            'epoch': epoch,
            'ff_accuracy': ff_correct / total,
            'bp_accuracy': bp_correct / total,
            'total_samples': total,
            'ff_predictions': ff_predictions,
            'bp_predictions': bp_predictions,
            'true_labels': true_labels
        }

    def create_tsne_visualization(self, ff_activations, bp_activations, labels, epoch, save_path):
        """创建t-SNE可视化"""
        try:
            # 选择最后一层的激活值
            ff_features = ff_activations[-1].view(ff_activations[-1].size(0), -1).cpu().numpy()
            bp_features = bp_activations[-1].view(bp_activations[-1].size(0), -1).cpu().numpy()
            
            # 降维到合适的维度
            if ff_features.shape[1] > 50:
                pca = PCA(n_components=50)
                ff_features = pca.fit_transform(ff_features)
                bp_features = pca.transform(bp_features)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ff_features)-1))
            ff_tsne = tsne.fit_transform(ff_features)
            bp_tsne = tsne.fit_transform(bp_features)
            
            # 可视化
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(ff_tsne[:, 0], ff_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.title(f'FF Features t-SNE (Epoch {epoch})')
            plt.colorbar(scatter)
            
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(bp_tsne[:, 0], bp_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.title(f'BP Features t-SNE (Epoch {epoch})')
            plt.colorbar(scatter)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'tsne_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")

    def save_comparison_results(self, all_results, filename='comparison_results_v2.json'):
        """保存比较结果"""
        # 移除numpy数组以便JSON序列化
        results_copy = all_results.copy()
        if 'performance_comparisons' in results_copy:
            for comp in results_copy['performance_comparisons']:
                for key in ['ff_predictions', 'bp_predictions', 'true_labels']:
                    if key in comp:
                        del comp[key]
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        print(f"Comparison results saved to {filepath}")

    def plot_comprehensive_comparisons(self, results):
        """绘制综合比较图"""
        epochs = [res['epoch'] for res in results['performance_comparisons']]
        
        plt.figure(figsize=(20, 15))
        
        # 1. 准确率比较
        plt.subplot(3, 3, 1)
        plt.plot(epochs, [res['ff_accuracy'] for res in results['performance_comparisons']], 
                label='Forward-Forward', linewidth=2)
        plt.plot(epochs, [res['bp_accuracy'] for res in results['performance_comparisons']], 
                label='Backpropagation', linewidth=2)
        plt.title('Accuracy Comparison', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 损失比较
        plt.subplot(3, 3, 2)
        plt.plot(epochs, results['ff_losses'], label='FF Loss', linewidth=2)
        plt.plot(epochs, results['bp_losses'], label='BP Loss', linewidth=2)
        plt.title('Loss Comparison', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. 能耗比较
        plt.subplot(3, 3, 3)
        plt.plot(epochs, results['ff_power_usage'], label='FF Power Usage', linewidth=2)
        plt.plot(epochs, results['bp_power_usage'], label='BP Power Usage', linewidth=2)
        plt.title('Power Usage Comparison', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Power Usage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. 内存使用比较
        plt.subplot(3, 3, 4)
        plt.plot(epochs, results['ff_memory_usage'], label='FF Memory', linewidth=2)
        plt.plot(epochs, results['bp_memory_usage'], label='BP Memory', linewidth=2)
        plt.title('Memory Usage Comparison', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. 训练时间比较
        plt.subplot(3, 3, 5)
        plt.plot(epochs, results['ff_training_time'], label='FF Training Time', linewidth=2)
        plt.plot(epochs, results['bp_training_time'], label='BP Training Time', linewidth=2)
        plt.title('Training Time Comparison', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. 权重相似度
        if results['weight_comparisons']:
            plt.subplot(3, 3, 6)
            num_layers = len(results['weight_comparisons'][0])
            for i in range(num_layers):
                similarities = [res[i]['weight_cosine_sim'] for res in results['weight_comparisons']]
                plt.plot(epochs, similarities, label=f'Layer {i}', linewidth=2)
            plt.title('Weight Cosine Similarity', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('Similarity')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 7. 激活值相关性
        if results['activation_comparisons']:
            plt.subplot(3, 3, 7)
            num_layers = len(results['activation_comparisons'][0])
            for i in range(num_layers):
                correlations = [res[i]['activation_correlation'] for res in results['activation_comparisons']]
                plt.plot(epochs, correlations, label=f'Layer {i}', linewidth=2)
            plt.title('Activation Correlation', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # 8. FLOPs比较
        plt.subplot(3, 3, 8)
        plt.bar(['FF', 'BP'], [results['ff_flops'], results['bp_flops']], 
                color=['blue', 'red'], alpha=0.7)
        plt.title('FLOPs Comparison', fontsize=14)
        plt.ylabel('FLOPs')
        plt.grid(True, alpha=0.3)

        # 9. 准确率差异
        plt.subplot(3, 3, 9)
        acc_diff = [abs(ff - bp) for ff, bp in zip(
            [res['ff_accuracy'] for res in results['performance_comparisons']],
            [res['bp_accuracy'] for res in results['performance_comparisons']]
        )]
        plt.plot(epochs, acc_diff, color='green', linewidth=2)
        plt.title('Accuracy Difference', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('|FF_Acc - BP_Acc|')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'comprehensive_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_models(architecture, dataset, num_classes, img_shape):
    """创建模型"""
    if architecture == 'fc':
        img_size = np.prod(img_shape)
        dims = [img_size, 500, 500]
        ff_net = FFNet_V2(dims, num_classes)
        bp_net = BPNet_V2(dims, num_classes)
    elif architecture == 'lenet':
        ff_net = LeNet5_FF(num_classes)
        bp_net = LeNet5_BP(num_classes)
    elif architecture == 'resnet':
        ff_net = ResNet101_FF(num_classes)
        bp_net = ResNet101_BP(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return ff_net, bp_net

def main():
    parser = argparse.ArgumentParser(description='Advanced FF vs BP Comparison v2')
    parser.add_argument('--architecture', type=str, default='fc', 
                       choices=['fc', 'lenet', 'resnet'], help='Model architecture')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10', 'cifar100'], help='Dataset')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Test batch size')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--comparison-dir', type=str, default='./comparison_results_v2', 
                       help='Directory to save results')
    parser.add_argument('--save-tsne', action='store_true', help='Save t-SNE visualizations')
    parser.add_argument('--tsne-interval', type=int, default=50, help='t-SNE saving interval')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print(f"Starting experiment: {args.architecture} on {args.dataset}")
    print("Loading data...")
    
    # 选择数据加载函数
    if args.architecture == 'fc':
        train_loader, test_loader, num_classes, img_shape = get_loaders(
            args.dataset, args.train_batch_size, args.test_batch_size, args.data_dir)
    else:
        train_loader, test_loader, num_classes, img_shape = get_loaders_v2(
            args.dataset, args.train_batch_size, args.test_batch_size, args.data_dir)
    
    print(f"Data loaded: {num_classes} classes, image shape: {img_shape}")
    
    # 创建模型
    print("Creating models...")
    ff_net, bp_net = create_models(args.architecture, args.dataset, num_classes, img_shape)
    
    # 编译模型
    if not args.no_compile:
        print("Compiling models...")
        try:
            ff_net = torch.compile(ff_net)
            bp_net = torch.compile(bp_net)
            print("Models compiled successfully")
        except Exception as e:
            print(f"Error compiling models: {e}")
    
    # 初始化比较器
    comparator = AdvancedNetworkComparator(save_dir=args.comparison_dir)
    
    # 计算FLOPs
    print("Computing FLOPs...")
    ff_flops = comparator.compute_flops(ff_net, img_shape)
    bp_flops = comparator.compute_flops(bp_net, img_shape)
    
    print(f"FF FLOPs: {ff_flops:,}")
    print(f"BP FLOPs: {bp_flops:,}")
    
    # 初始化结果记录
    all_results = {
        'ff_losses': [], 'bp_losses': [],
        'ff_power_usage': [], 'bp_power_usage': [],
        'ff_memory_usage': [], 'bp_memory_usage': [],
        'ff_training_time': [], 'bp_training_time': [],
        'weight_comparisons': [], 'activation_comparisons': [],
        'performance_comparisons': [],
        'ff_flops': ff_flops, 'bp_flops': bp_flops,
        'experiment_info': {
            'architecture': args.architecture,
            'dataset': args.dataset,
            'num_epochs': args.num_epochs,
            'batch_size': args.train_batch_size,
            'img_shape': img_shape,
            'num_classes': num_classes
        }
    }

    print(f"Starting training for {args.num_epochs} epochs...")
    
    # 获取一批数据用于训练
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    
    # 创建正负样本
    if args.architecture == 'fc':
        # 对于全连接网络，使用原始的overlay方法
        x_pos = overlay_y_on_x(x, y, num_classes)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd], num_classes)
        y_neg = y[rnd]
    else:
        # 对于CNN，使用新的overlay方法
        x_pos = overlay_y_on_x_v2(x, y, num_classes)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x_v2(x, y[rnd], num_classes)
        y_neg = y[rnd]
    
    # 训练循环
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # FF训练
        print("Training FF network...")
        comparator.power_monitor.start_monitoring()
        ff_start_time = time.time()
        
        if args.architecture in ['lenet', 'resnet']:
            ff_loss = ff_net.train_epoch(x_pos, x_neg, y, y_neg)
        else:
            ff_loss = ff_net.train_epoch(x_pos, x_neg)
        
        ff_end_time = time.time()
        comparator.power_monitor.stop_monitoring()
        ff_power_usage = comparator.power_monitor.get_average_usage()
        
        # BP训练
        print("Training BP network...")
        comparator.power_monitor = PowerMonitor()  # 重新初始化
        comparator.power_monitor.start_monitoring()
        bp_start_time = time.time()
        
        x_combined = torch.cat([x_pos, x_neg], dim=0)
        y_combined = torch.cat([y, y_neg], dim=0)
        
        bp_loss = bp_net.train_epoch(x_combined, y_combined)
        
        bp_end_time = time.time()
        comparator.power_monitor.stop_monitoring()
        bp_power_usage = comparator.power_monitor.get_average_usage()
        
        # 记录结果
        all_results['ff_losses'].append(ff_loss)
        all_results['bp_losses'].append(bp_loss)
        all_results['ff_power_usage'].append(ff_power_usage['gpu_avg'])
        all_results['bp_power_usage'].append(bp_power_usage['gpu_avg'])
        all_results['ff_memory_usage'].append(ff_power_usage['gpu_memory_avg'])
        all_results['bp_memory_usage'].append(bp_power_usage['gpu_memory_avg'])
        all_results['ff_training_time'].append(ff_end_time - ff_start_time)
        all_results['bp_training_time'].append(bp_end_time - bp_start_time)
        
        # 网络比较
        ff_layer_info = ff_net.get_layer_info()
        bp_layer_info = bp_net.get_layer_info()
        
        weight_comp = comparator.compare_weights(ff_layer_info, bp_layer_info, epoch)
        all_results['weight_comparisons'].append(weight_comp)
        
        # 激活值比较
        with torch.no_grad():
            _, ff_activations = ff_net.forward_with_activations(x_pos)
            _, bp_activations = bp_net.forward_with_activations(x_combined)
        
        activation_comp = comparator.compare_activations(ff_activations, bp_activations, epoch)
        all_results['activation_comparisons'].append(activation_comp)
        
        # 性能比较
        performance_comp = comparator.compare_performance(ff_net, bp_net, test_loader, epoch)
        all_results['performance_comparisons'].append(performance_comp)
        
        # 打印进度
        print(f"FF Loss: {ff_loss:.4f}, BP Loss: {bp_loss:.4f}")
        print(f"FF Acc: {performance_comp['ff_accuracy']:.4f}, BP Acc: {performance_comp['bp_accuracy']:.4f}")
        print(f"FF Time: {ff_end_time - ff_start_time:.2f}s, BP Time: {bp_end_time - bp_start_time:.2f}s")
        
        # 保存t-SNE可视化
        if args.save_tsne and (epoch + 1) % args.tsne_interval == 0:
            try:
                comparator.create_tsne_visualization(
                    ff_activations, bp_activations, 
                    performance_comp['true_labels'][:100],  # 只使用前100个样本
                    epoch + 1, args.comparison_dir
                )
            except Exception as e:
                print(f"Error saving t-SNE: {e}")
    
    print("\nTraining completed!")
    print("Saving results...")
    
    # 保存模型
    torch.save(ff_net.state_dict(), os.path.join(args.comparison_dir, 'ff_net_final.pth'))
    torch.save(bp_net.state_dict(), os.path.join(args.comparison_dir, 'bp_net_final.pth'))
    
    # 保存比较结果
    comparator.save_comparison_results(all_results)
    
    # 绘制综合比较图
    print("Creating comprehensive plots...")
    try:
        comparator.plot_comprehensive_comparisons(all_results)
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    # 打印总结
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Architecture: {args.architecture}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.num_epochs}")
    print(f"FF FLOPs: {ff_flops:,}")
    print(f"BP FLOPs: {bp_flops:,}")
    
    final_ff_acc = all_results['performance_comparisons'][-1]['ff_accuracy']
    final_bp_acc = all_results['performance_comparisons'][-1]['bp_accuracy']
    print(f"Final FF Accuracy: {final_ff_acc:.4f}")
    print(f"Final BP Accuracy: {final_bp_acc:.4f}")
    print(f"Accuracy Difference: {abs(final_ff_acc - final_bp_acc):.4f}")
    
    avg_ff_time = np.mean(all_results['ff_training_time'])
    avg_bp_time = np.mean(all_results['bp_training_time'])
    print(f"Average FF Training Time: {avg_ff_time:.2f}s")
    print(f"Average BP Training Time: {avg_bp_time:.2f}s")
    
    print(f"\nResults saved to: {args.comparison_dir}")

if __name__ == "__main__":
    main()