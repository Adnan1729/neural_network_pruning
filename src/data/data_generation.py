def generate_dummy_data(n_samples=1000, n_features=20):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, (n_samples, 1))
    return X, y
