def create_pruning_mask(model, pruning_threshold):
    masks = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            weights = layer.get_weights()[0]
            mask = np.abs(weights) > pruning_threshold
            masks.append(mask)
        else:
            masks.append(None)
    return masks
