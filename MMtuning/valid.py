import torch
import json
import os
import json
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from rouge import Rouge
from tqdm import tqdm
from .blip2_mmtuning import Blip2MMtuning
from .VQA_dataload import VQADataset

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
image_folder = "val" 

merged_data_val = {}
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

                    merged_data_val[folder_name] = {
                        "prompt": text_input,
                        "topic": topic,
                        "text_input": "",
                        "text_output": text_output,
                        "image_path": image_path,  
                    }




model= Blip2MMtuning(vit_model='google/vit-large-patch16-224', 
                            t5_model='t5-3b', vit_precision="fp32",
                            freeze_vit=True, llm_peft=True, freeze_qformer=True, 
                            dtype_llm = torch.bfloat16)
lora_parameters = model.state_dict()
trainable_parameters_pretrained = torch.load('result/checkpoint_0.pth')
pretrained_weights = torch.load('pretrained/blip2_pretrained.pth')
pretrained_parameters = pretrained_weights['model']

for key in pretrained_parameters:
    if key in lora_parameters:
        lora_parameters[key] = pretrained_parameters[key]    

for key in trainable_parameters_pretrained :
    if key in lora_parameters:
        lora_parameters[key] = trainable_parameters_pretrained[key]    

model.load_state_dict(lora_parameters)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
VQA_dataset_val = VQADataset(merged_data_val, transform=transform)

batch_size = 3
val_loader = DataLoader(VQA_dataset_val, batch_size=batch_size, shuffle=True)
rouge = Rouge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

val_loss = 0
rouge1_scores_for_samples = []
rougeL_scores_for_samples = []

unique_topic_values = {'capitalization', 'literacy-in-science', 'global-studies', 'us-history', 
                       'writing-strategies', 'phonological-awareness', 'chemistry', 'world-history', 
                       'units-and-measurement', 'biology', 'grammar', 'vocabulary', 'physics', 
                       'earth-science', 'science-and-engineering-practices', 'culture', 
                       'reading-comprehension', 'reference-skills', 'figurative-language', 
                       'geography', 'word-study', 'verbs', 'economics', 'punctuation', 
                       'pronouns', 'civics'}
accuracy_by_topic = {topic: {"total_samples": 0, "correct_samples": 0} for topic in unique_topic_values}

model.eval()
with torch.no_grad(), tqdm(total=len(val_loader)) as pbar:
    for _, sample in enumerate(val_loader):
        
        outputs, loss = model(sample)
        generated_text_lora_inference = model.generate(sample)

        for generated_sample, reference_sample, topic in zip(generated_text_lora_inference, sample["text_output"], sample["topic"]):
                if generated_sample.strip() == '' or len(generated_sample.strip())<=0 or generated_sample.strip() == "." :
                    generated_sample = 'The answer:'
                rouge_scores = rouge.get_scores(generated_sample, reference_sample, avg=True) 
                rouge1_scores_for_samples.append(rouge_scores['rouge-1']['f'])
                rougeL_scores_for_samples.append(rouge_scores['rouge-l']['f'])
                accuracy_by_topic[topic]["total_samples"] += 1
                if generated_sample == reference_sample:
                    accuracy_by_topic[topic]["correct_samples"] += 1
        
        val_loss += loss.item()
        pbar.update(1) 

    total_batch = len(val_loader)
    val_loss_epoch = val_loss/ total_batch
    validation_rouge1_score = sum(rouge1_scores_for_samples) / len(rouge1_scores_for_samples)
    validation_rougeL_score = sum(rougeL_scores_for_samples) / len(rougeL_scores_for_samples)
    
    print(f"Val_loss:{val_loss_epoch}")
    print(f"ROUGE-1 Score:{validation_rouge1_score}")
    print(f"ROUGE-L Score:{validation_rougeL_score}")
    print("\nGenerated text accuracy categorized by TOPIC:")
    for topic, accuracy_data in accuracy_by_topic.items():
        if accuracy_data["total_samples"] == 0:
            print(f"{topic}: This topic is not tested")
            continue
        accuracy_percentage = (accuracy_data["correct_samples"] / accuracy_data["total_samples"]) * 100
        print(f"{topic}: {accuracy_percentage:.3f}% ({accuracy_data['correct_samples']}/{accuracy_data['total_samples']})")
