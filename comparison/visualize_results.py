import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_weights(model_path, title, img_size, num_filters=10, save_path=None):
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 找到第一层的权重
    first_layer_weight = None
    for key, value in model_state_dict.items():
        if 'layers.0.weight' in key:
            first_layer_weight = value.cpu().numpy()
            break
    
    if first_layer_weight is None:
        print(f"Could not find first layer weights in {model_path}")
        return

    # 假设输入是扁平化的图像，将其重塑为正方形
    side = int(np.sqrt(img_size))
    if side * side != img_size:
        print(f"Warning: img_size {img_size} is not a perfect square. Visualization might be distorted.")
        # Fallback for non-square images, e.g., CIFAR (3*32*32)
        if img_size == 3 * 32 * 32: # CIFAR10/100
            num_channels = 3
            side = 32
            # Reshape for CIFAR: (out_features, in_channels * height * width)
            # Need to reshape to (out_features, in_channels, height, width) for visualization
            first_layer_weight = first_layer_weight.reshape(first_layer_weight.shape[0], num_channels, side, side)
        else:
            num_channels = 1 # Default to grayscale

    num_filters_to_show = min(num_filters, first_layer_weight.shape[0])
    
    fig, axes = plt.subplots(1, num_filters_to_show, figsize=(num_filters_to_show * 2, 2))
    if num_filters_to_show == 1:
        axes = [axes]

    for i in range(num_filters_to_show):
        filter_weights = first_layer_weight[i]
        
        if num_channels == 3: # CIFAR
            # Transpose to (height, width, channels) for imshow
            filter_weights = np.transpose(filter_weights, (1, 2, 0))
            # Normalize to [0, 1] for proper display
            filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min())
            axes[i].imshow(filter_weights)
        else: # MNIST
            filter_weights = filter_weights.reshape(side, side)
            axes[i].imshow(filter_weights, cmap='gray')
        
        axes[i].axis('off')
        axes[i].set_title(f'Filter {i+1}')
    
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.close(fig) # Close the figure to free memory

def main():
    comparison_dir = './comparison_results_v3'
    results_file = os.path.join(comparison_dir, 'comparison_results.json')

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    # Re-plot comparisons
    comparator = NetworkComparator(save_dir=comparison_dir) # Re-initialize comparator for plotting
    comparator.plot_comparisons(all_results)

    # Visualize weights
    img_size = all_results['training_info']['img_size']

    # Final weights
    visualize_weights(
        os.path.join(comparison_dir, 'ff_net_final.pth'), 
        'FFNet Final Layer 0 Weights',
        img_size,
        save_path=os.path.join(comparison_dir, 'ff_final_weights.png')
    )
    visualize_weights(
        os.path.join(comparison_dir, 'bp_net_final.pth'), 
        'BPNet Final Layer 0 Weights',
        img_size,
        save_path=os.path.join(comparison_dir, 'bp_final_weights.png')
    )

    # Checkpoint weights (e.g., epoch 100)
    visualize_weights(
        os.path.join(comparison_dir, 'ff_net_epoch_100.pth'), 
        'FFNet Epoch 100 Layer 0 Weights',
        img_size,
        save_path=os.path.join(comparison_dir, 'ff_epoch_100_weights.png')
    )
    visualize_weights(
        os.path.join(comparison_dir, 'bp_net_epoch_100.pth'), 
        'BPNet Epoch 100 Layer 0 Weights',
        img_size,
        save_path=os.path.join(comparison_dir, 'bp_epoch_100_weights.png')
    )

# Dummy NetworkComparator for plotting
class NetworkComparator:
    def __init__(self, save_dir):
        self.save_dir = save_dir

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
        plt.close() # Close the figure to free memory

if __name__ == '__main__':
    main()