import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import os
from torch.cuda.amp import autocast
from downstream_model import PartModel, ACModel, GradeModel, load_and_transform_image
import pandas as pd

# Load models
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

Encoder = model.visual_encoder

PartModel = PartModel()
checkpoint_file = "output/GIRepoter/Clspart_freeze_vitg_ep20-1.pth"
checkpoint = torch.load(checkpoint_file, map_location='cpu')
res = PartModel.load_state_dict(checkpoint, strict=False)
PartModel = PartModel.to(device)

ACModel = ACModel()
checkpoint_file = "output/GIRepoter/ours_bs64_lr3e-4_0213-002-8.pth"
checkpoint = torch.load(checkpoint_file, map_location='cpu')
res = ACModel.load_state_dict(checkpoint, strict=False)
ACModel = ACModel.to(device)

GradeModel = GradeModel()
checkpoint_file = "output/GIRepoter/Grad_freeze_vitg_ep30-1.pth"
checkpoint = torch.load(checkpoint_file, map_location='cpu')
res = GradeModel.load_state_dict(checkpoint, strict=False)
GradeModel = GradeModel.to(device)

Encoder.eval()
PartModel.eval()
ACModel.eval()
GradeModel.eval()

# Anatomical Region Classification & Abnormal State Recognition
case_path = "sample_data/GIRepoter_sample_data/case_2"
case_list = [os.path.join(case_path, file) for file in os.listdir(case_path) if file.endswith('.jpg') or file.endswith('.png')]
batchsize = 32

all_results = []
part_names = ["Esophagus", "Cardia", "Gastric Fundus", "Gastric Body", "Gastric Angle", "Gastric Antrum and Pylorus", "Duodenum"]
part_names_chinese = ["食道", "贲门", "胃底", "胃体", "胃角", "胃窦和幽门", "十二指肠"]

# Anatomical Region Classification & Abnormal State Recognition
for i in range(0, len(case_list), batchsize):
    batch_files = case_list[i:i+batchsize]

    batch_images = torch.stack([
        load_and_transform_image(file).to(device) for file in batch_files
    ])

    with torch.no_grad():
        with autocast():
            feats = Encoder(batch_images)

            parts_outputs = PartModel(feats)
            parts_probs = torch.softmax(parts_outputs, dim=1)
            _, parts_predicted = torch.max(parts_probs.data, 1)

            part_confidences = parts_probs[range(len(parts_probs)), parts_predicted]

            acs_outputs = ACModel(feats)
            ac_probs = torch.softmax(acs_outputs, dim=1)
            anomaly_scores = ac_probs[:, 1]

    for file, part, ac_score, conf in zip(
        batch_files,
        parts_predicted.cpu().numpy(),
        anomaly_scores.cpu().numpy(),
        part_confidences.cpu().numpy()
    ):
        all_results.append({
            "filename": os.path.basename(file),
            "filepath": file,
            "part_id": int(part),
            "part_name": part_names[part],
            "anomaly_score": float(ac_score),
            "part_confidence": float(conf)
        })

df = pd.DataFrame(all_results)
grouped = df.groupby("part_name", sort=False)

images_description = {part_name: [] for part_name in part_names}
images_conclusion = {part_name: [] for part_name in part_names}

for part_name, group in grouped:
    sorted_group = group.sort_values(by="anomaly_score", ascending=False)
    for idx, (_, row) in enumerate(sorted_group.iterrows()):
        if len(images_description[part_name]) < 10:
            images_description[part_name].append(row['filepath'])
        if (part_name == "Gastric Antrum and Pylorus" and idx < 2) or \
           (part_name != "Gastric Antrum and Pylorus" and idx < 1):
            images_conclusion[part_name].append(row['filepath'])

print("Images for Region Description Generation:")
for part_name, images in images_description.items():
    print(f"{part_name}: {len(images)} images - {', '.join(images)}")

print("\nImages for Diagnosis Conclusion Generation:")
for part_name, images in images_conclusion.items():
    print(f"{part_name}: {len(images)} images - {', '.join(images)}")

# Region Description Generation
region_descriptions = []
for region, images in images_description.items():
    images = [[vis_processors["eval"](Image.open(image).convert("RGB")).to(device) for image in images]]
    response = model.generate({"image": images, "prompt": 'Please describe this set of gastroscopic images from the same gastric region.'})
    region_descriptions.append(response[0])
    print(f"Region: {region}")
    print(f"Description: {response[0]}")
    print('-' * 20)

# Diagnosis Text Generation
images = [[vis_processors["eval"](Image.open(image).convert("RGB")).to(device) for sublist in images_conclusion.values() for image in sublist]]
response = model.generate({"image": images, "prompt": 'Please draw a diagnostic conclusion from this set of gastroscopy images.'})
conclusion = response[0]
print(f"Conclusion: {response[0]}")

# Upper GI Cancer Screening 
grade = {0: "high-risk", 1: "low-risk"}
grade_chinese = {0: "高风险", 1: "低风险"}

batch_images = torch.stack([
    load_and_transform_image(image).to(device) for sublist in images_conclusion.values() for image in sublist
])

# Upper GI Cancer Screening 
with torch.no_grad():
    with autocast():
        feats = Encoder(batch_images)
        grade_logits = GradeModel(feats)
        grade_probs = torch.softmax(grade_logits, dim=0)
        grade_pred = torch.argmax(grade_probs).item()
        print(f"Result of Upper GI Cancer Screening: {grade[grade_pred]}")

# Create Reports
from create_report import create_report_english, create_report_chinese
create_report_chinese(region_descriptions, conclusion, images_conclusion, part_names, part_names_chinese, grade_chinese, grade_pred)
create_report_english(region_descriptions, conclusion, images_conclusion, part_names, part_names_chinese, grade, grade_pred)
