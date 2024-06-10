import torch
import json
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .VQA_dataload import VQADataset
from .blip2_mmtuning import Blip2MMtuning

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Please specify the folder for the dataset
data_folder = "problems.json"
image_folder = "train"  

merged_data_train = {}
for folder_name in tqdm(os.listdir(image_folder)):
    if os.path.isdir(os.path.join(image_folder, folder_name)):
        with open(data_folder, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            if data[folder_name]['image'] is not None:
                image_folder_path = os.path.join(image_folder, folder_name)
                image_filename = data[folder_name]['image']
                
                image_path = []
                if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
                    image_paths = os.path.join(image_folder_path, image_filename)
                    image_path.append(image_paths)
                        
                    text_input = 'question:' + " " + data[folder_name]['question'] + " " + 'context:' + " " + data[folder_name]['hint'] + "\n " + "choices: " + ', '.join(data[folder_name]['choices']) + " " + "Task: " + "Please select correct description of the answer from the 'choices'" + " " + "answer:"
                    text_output = data[folder_name]['choices'][data[folder_name]['answer']]
                    topic = data[folder_name]['topic']

                    merged_data_train[folder_name] = {
                        "prompt": "",
                        "topic": topic,
                        "text_input": text_input,
                        "text_output": text_output,
                        "image_path": image_path, 
                    }


model = Blip2MMtuning(vit_model='google/vit-large-patch16-224', 
                            t5_model='t5-3b', vit_precision="fp32",
                            freeze_vit=True, vit_peft=False, llm_peft=True, freeze_qformer=True, 
                            dtype_llm = torch.bfloat16)
model_parameters = model.state_dict()
pretrained_weights = torch.load('pretrained/blip2_pretrained.pth')
pretrained_parameters = pretrained_weights['model']
for key in pretrained_parameters:
    if key in model_parameters:
        model_parameters[key] = pretrained_parameters[key]

model.load_state_dict(model_parameters)
for name, param in model.named_parameters():
    if ('lora_A_1' in name or 'lora_B_1' in name or 'lora_A_2' in name or 'lora_B_2' in name
            or 'lora_A_3' in name or 'lora_B_3' in name or 'lora_A_4' in name or 'lora_B_4' in name
            or 'router_img' in name or 'router_text' in name or 'router_single' in name):
        param.requires_grad_()
 


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
batch_size = 8

VQA_dataset_train = VQADataset(merged_data_train, transform=transform)
train_loader = DataLoader(VQA_dataset_train, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer_llm= AdamW(model.parameters(), lr=1e-5, weight_decay=1e-8)

epochs = 3
total_steps = len(train_loader) * epochs
scheduler_llm= get_cosine_schedule_with_warmup(optimizer_llm, num_warmup_steps = 4000, num_training_steps = total_steps)

train_losses = []
for epoch in range(epochs):
    train_loss = 0
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as t:
        for _, sample in t:
            outputs, loss = model(sample)
            optimizer_llm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer_llm.step()
            scheduler_llm.step()
            train_loss += loss.item()
            t.set_postfix(loss=train_loss / (t.n + 1))

    optimizer_llm.zero_grad()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f"Epoch: {epoch}, Loss: {train_loss}")
    trainable_parameters = {param_name: param for param_name, param in model.named_parameters() if param.requires_grad}
    torch.save(trainable_parameters, f'result/checkpoint_{epoch}.pth')
