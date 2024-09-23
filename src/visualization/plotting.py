# Visualization Functions
def plot_weight_distribution(weights_list, title):
    plt.figure(figsize=(10, 6))
    for i, weights in enumerate(weights_list):
        if weights is not None:
            plt.hist(weights.flatten(), bins=50, alpha=0.5, label=f'Layer {i+1}')
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_accuracy_comparison(history, fine_tune_history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Original Training')
    plt.plot(history.history['val_accuracy'], label='Original Validation')
    plt.plot(range(10, 15), fine_tune_history.history['accuracy'], label='Fine-tuned Training')
    plt.plot(range(10, 15), fine_tune_history.history['val_accuracy'], label='Fine-tuned Validation')
    plt.title('Model Accuracy: Original vs Fine-tuned')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_sparsity(original_weights, pruned_weights):
    original_sparsity = [np.sum(w == 0) / w.size if w is not None else 0 for w in original_weights]
    pruned_sparsity = [np.sum(w == 0) / w.size if w is not None else 0 for w in pruned_weights]
    
    layers = [f'Layer {i+1}' for i in range(len(original_sparsity))]
    x = range(len(layers))
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], original_sparsity, width=0.4, label='Original', align='center')
    plt.bar([i + 0.2 for i in x], pruned_sparsity, width=0.4, label='Pruned', align='center')
    plt.title('Layer-wise Sparsity: Original vs Pruned')
    plt.xlabel('Layer')
    plt.ylabel('Sparsity')
    plt.xticks(x, layers)
    plt.legend()
    plt.show()
