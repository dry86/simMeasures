import numpy as np

def disagreement(predictions1, predictions2):
    # Ensure the inputs are numpy arrays for easy comparison
    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)
    
    # Calculate the disagreement by comparing the argmax of both predictions
    disagreement_rate = np.mean(np.argmax(predictions1, axis=1) != np.argmax(predictions2, axis=1))
    
    return disagreement_rate

# Example predictions (class probabilities for 3 inputs, 3 classes)
model1_preds = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.7, 0.2, 0.1]])
model2_preds = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1]])

# Calculate disagreement
disagreement_rate = disagreement(model1_preds, model2_preds)
print(f"Disagreement Rate: {disagreement_rate}")


