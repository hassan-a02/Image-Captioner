from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.optim as optim
import pickle
import os
from torchvision import transforms
from model import ImageCaptioner, Vocabulary
from flask import send_from_directory


app = Flask(__name__)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


vocab_path = os.path.join(os.path.dirname(__file__), "model/vocab.pkl")


with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptioner(256, 256, len(vocab), 1).to(device)
optimiser = optim.Adam(model.parameters(), lr=3e-4)

weights_path = os.path.join(os.path.dirname(__file__), "model", "best_weights.pth.tar")
step = load_checkpoint(torch.load(weights_path, weights_only=False), model, optimiser)
model.eval()

transform = transforms.Compose([
    transforms.Resize((356, 356)),
    transforms.CenterCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    caption_tokens = model.gen_caption(image, vocab)


    ignore_tokens = ["<SOS>", "<EOS>", "<PAD>"]
    caption = ' '.join(
    [token for token in caption_tokens if token not in ignore_tokens])
    return jsonify({'caption': caption})


@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../static', path)


if __name__ == '__main__':
    app.run(debug=True)
