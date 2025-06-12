import numpy as np
import onnxruntime as ort


session = ort.InferenceSession("models/titanic_model.onnx")

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name


def predict_survival(input_features):
    
    input_array = np.array(input_features, dtype=np.float32).reshape(1, -1)
    
    output = session.run([output_name], {input_name: input_array})[0]
    
    probability = float(output[0][0])
    
    prediction = int(probability > 0.5)
    
    return prediction

if __name__ == "__main__":
    sample = [3, 0, 26.0, 0, 0, 7.92, 2]    
    
    prediction = predict_survival(sample)
    
    print(f"Prediction Class: {prediction}")