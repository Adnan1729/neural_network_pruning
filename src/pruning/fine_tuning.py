def get_model_weights(model):
    return [layer.get_weights()[0] if isinstance(layer, keras.layers.Dense) else None 
            for layer in model.layers]

def calculate_sparsity(weights_list):
    total_params = sum(w.size for w in weights_list if w is not None)
    zero_params = sum(np.sum(w == 0) for w in weights_list if w is not None)
    return zero_params / total_params
