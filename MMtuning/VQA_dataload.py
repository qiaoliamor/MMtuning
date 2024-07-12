import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from PIL import Image
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

class VQADataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.index_mapping = {idx: key for idx, key in enumerate(data_dict.keys())}

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        key = self.index_mapping[idx]
        sample = self.data_dict[key]
        images_paths = sample["image_path"]
        text_input = sample["text_input"]
        text_output = sample["text_output"]
        prompt = sample["prompt"]
        topics = sample['topic']

        # 如果有 transform 函数，则对图像进行预处理
        images = []
        for img_pth in images_paths:
            with Image.open(img_pth) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                images.append(image)

        if len(images)>1:
            cated_images = torch.cat(images, dim=-1)
            cated_images = F.resize(cated_images, (224, 224))
        else:
            cated_images = images[0]
        
        return {
            "image": cated_images, 
            "prompt": prompt,
            "text_input": text_input,
            "text_output": text_output,
            "topic": topics,
        }
