# Main Execution
def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Generate data
    X, y = generate_dummy_data()

    # Create and train model
    model = create_model(input_shape=(20,))
    history = train_model(model, X, y)

    # Evaluate original model
    loss, accuracy = evaluate_model(model, X, y)
    print(f"Before pruning - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Pruning process
    original_weights = get_model_weights(model)
    plot_weight_distribution(original_weights, 'Weight Distribution Before Pruning')

    pruning_threshold = 0.1
    pruning_masks = create_pruning_mask(model, pruning_threshold)

    # Visualize theoretical pruning
    pruned_weights = [w * m if w is not None and m is not None else None 
                      for w, m in zip(original_weights, pruning_masks)]
    plot_weight_distribution(pruned_weights, 'Weight Distribution After Pruning')

    # Calculate and print sparsity
    original_sparsity = calculate_sparsity(original_weights)
    pruned_sparsity = calculate_sparsity(pruned_weights)
    print(f"Original sparsity: {original_sparsity:.2%}")
    print(f"Pruned sparsity: {pruned_sparsity:.2%}")

    # Apply pruning to the model
    apply_pruning_mask(model, pruning_masks)

    # Evaluate pruned model
    loss, accuracy = evaluate_model(model, X, y)
    print(f"After pruning - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Fine-tune the pruned model
    fine_tune_history = train_model(model, X, y, epochs=5)

    # Evaluate fine-tuned model
    loss, accuracy = evaluate_model(model, X, y)
    print(f"After fine-tuning - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Final visualizations
    final_weights = get_model_weights(model)
    plot_weight_distribution(final_weights, 'Weight Distribution After Fine-tuning')
    plot_accuracy_comparison(history, fine_tune_history)
    plot_sparsity(original_weights, final_weights)

    # Calculate final sparsity
    final_sparsity = calculate_sparsity(final_weights)
    print(f"Final sparsity: {final_sparsity:.2%}")

if __name__ == "__main__":
    main()
