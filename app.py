import io
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torch import nn

app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=5*1024*1024
ckpt_path=Path(__file__).parent/'saved_models/waste_model_v3.pth'
if not ckpt_path.exists():
    raise FileNotFoundError('V3 model not found. Run: python3 train_v3_combined.py --epochs 8')
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=True)
classes=ckpt['classes']
model=models.mobilenet_v3_small(weights=None)
model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(classes))
model.load_state_dict(ckpt['state_dict'])
model.eval()
size=ckpt['image_size']
transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

@app.get('/')
def home(): return render_template('index.html')

@app.post('/predict')
def predict():
    image_file=request.files.get('image')
    if not image_file: return jsonify(error='Upload an image with form field image'),400
    try:
        with Image.open(io.BytesIO(image_file.read())) as img: image=img.convert('RGB')
    except (UnidentifiedImageError,OSError,ValueError): return jsonify(error='Invalid image'),400
    with torch.inference_mode():
        probs=torch.softmax(model(transform(image).unsqueeze(0)),dim=1)[0].tolist()
    idx=max(range(len(probs)),key=lambda i:probs[i])
    return jsonify(prediction=classes[idx],confidence=round(probs[idx]*100,2),probabilities={c:round(probs[i]*100,2) for i,c in enumerate(classes)},model='MobileNetV3-Small V3 (multi-dataset fine-tuning)')

if __name__=='__main__':app.run(debug=False)
