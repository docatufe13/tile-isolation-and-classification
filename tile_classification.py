import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import glob
import os
from sklearn.metrics.pairwise import cosine_similarity
import csv

# prebacivanje na NVidia GPU ako je dostupan
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# učitavanje pretreniranog ResNet18 modela
model = resnet18(pretrained=True)
# uklanjamo klasifikacioni sloj - koristimo samo feature extractor
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()


transform = transforms.Compose([
    #resnet očekuje ulazne slike veličine 224x224
    transforms.Resize((224, 224)),
    # konverzija slike u tensor zbog toga što PyTorch radi sa tenzorima
    transforms.ToTensor(),
    # normalizacija slike prema standardnim vrednostima za ImageNet dataset
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
#funckija za određivanje ključnih karakteristika slike
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    features = features.squeeze().cpu().numpy()
    features = features.flatten()  
    return features

# učitavanje referentnih (idealnih) slika i ekstrakcija feature-a
ref_front_feat = extract_features('./base_tiles/tile_front_larger.png')
ref_back_feat = extract_features('./base_tiles/tile_back_larger.png')
ref_combined = np.concatenate([ref_front_feat, ref_back_feat])

# učitavanje slika iz dataset-a
front_dir = './dataset/front/'
back_dir = './dataset/back/'

front_paths = sorted(glob.glob(os.path.join(front_dir, '*.png')))
back_paths = sorted(glob.glob(os.path.join(back_dir, '*.png')))


results = []

# prag za klasifikaciju (izabran empirijski)
similarity_threshold = 0.74

for front_path, back_path in zip(front_paths, back_paths):
    name = os.path.basename(front_path).replace('_napred_', '_')
    front_name = os.path.basename(front_path)
    back_name = os.path.basename(back_path)
    true_label = 'Zadovoljavajuć' if name.startswith('PK') else 'Nezadovoljavajuć'

    front_feat = extract_features(front_path)
    back_feat = extract_features(back_path)
    combined_feat = np.concatenate([front_feat, back_feat])

    similarity = cosine_similarity(
        combined_feat.reshape(1, -1),
        ref_combined.reshape(1, -1)
    )[0][0]

    pred_label = 'Zadovoljavajuć' if similarity >= similarity_threshold else 'Nezadovoljavajuć'

    print(f"{name} => Pred: {pred_label} | True: {true_label} | Similarity: {similarity:.4f}")
    results.append([front_name,back_name, pred_label, true_label, similarity])

# čuvanje rezultata u CSV fajl
with open("classification_results_resnet.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Front image","Back image", "Predicted", "True", "Similarity"])
    writer.writerows(results)
