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

def apply_pruning_mask(model, masks):
    for layer, mask in zip(model.layers, masks):
        if isinstance(layer, keras.layers.Dense) and mask is not None:
            weights, bias = layer.get_weights()
            pruned_weights = weights * mask
            layer.set_weights([pruned_weights, bias])

