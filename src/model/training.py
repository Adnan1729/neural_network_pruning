def train_model(model, X, y, epochs=10, batch_size=32, validation_split=0.2):
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                        validation_split=validation_split, verbose=0)
    return history

def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y, verbose=0)
    return loss, accuracy
