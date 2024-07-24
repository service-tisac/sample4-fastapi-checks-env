from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define environment variables
FLASK_PORT = int(os.getenv("FLASK_PORT", 8888))
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "simple_nn_checkpoint.pth")

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model from checkpoint
def load_model(checkpoint_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Define input data model
class InputData(BaseModel):
    input: List[float]

# Create FastAPI app
app = FastAPI()

# Load the model checkpoint
model = load_model(CHECKPOINT_PATH)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model_check")
def model_check():
    try:
        load_model(CHECKPOINT_PATH)
        return {"status": "model loaded successfully"}
    except Exception as e:
        return {"status": "model loading failed", "error": str(e)}

@app.post("/predict")
def predict(data: InputData):
    if len(data.input) != 10:
        raise HTTPException(status_code=400, detail="Input must be a list of 10 floats.")
    
    input_tensor = torch.tensor(data.input).float().unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    
    return {"output": output.squeeze().item()}

@app.get("/inference_check")
def inference_check():
    try:
        test_input = [0.0] * 10  # Dummy input data
        input_tensor = torch.tensor(test_input).float().unsqueeze(0)
        with torch.no_grad():
            _ = model(input_tensor)
        return {"status": "inference successful"}
    except Exception as e:
        return {"status": "inference failed", "error": str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=FLASK_PORT)