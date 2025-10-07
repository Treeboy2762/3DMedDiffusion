

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import json 
import os
from transformers import AutoModel, AutoTokenizer
import random

class Singleres_dataset(Dataset):
    def __init__(self, root_dir=None, resolution= [32,32,32], generate_latents= False, max_text_len=77):
        self.all_files = []
        self.resolution = resolution
        self.generate_latents = generate_latents
        self.max_text_len = max_text_len
        
        self.embeddings = np.load("/home/daesungk/CT-RATE/labels/train_embeddings_fp16.npy", mmap_mode="r")
        self.text_encoder = None
        self.tokenizer = None
        
        if root_dir.endswith('json'):
            with open(root_dir) as json_file:
                dataroots = json.load(json_file)

            for key,value in dataroots.items():
                self.base = key
                self.df = pd.read_csv(value)

            self.file_num = len(self.df)
            print(f"Total files : {self.file_num}")
    

    def _initialize_encoder(self):
        """This function will be called once by each worker process."""
        # Use CPU for text encoder to avoid CUDA multiprocessing issues
        self.device = torch.device('cpu')
        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", 
            trust_remote_code=True, 
            torch_dtype=torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", 
            trust_remote_code=True
        )
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        

    def __len__(self):
        return self.file_num


    def __getitem__(self, index):
        
        if self.generate_latents:
            file_path = list(self.all_files[index].items())[0][1]
            img_np = np.load(file_path)[np.newaxis, :]
            imageout = torch.from_numpy(img_np).float()
            imageout = imageout.transpose(1,3).transpose(2,3)
            return imageout, file_path
        else:

            row = self.df.iloc[index]
            file_path = os.path.join(self.base, row['VolumeName'])
            img_np = np.load(file_path)
            latent = torch.from_numpy(img_np).float()
            report_embedding = torch.tensor(self.embeddings[index], dtype=torch.float16)
            # findings = row['Findings_EN']
            # report_embedding = self._encode_text(findings)
            return latent, report_embedding, torch.tensor(self.resolution)/64.0
    
    
    def _encode_text(self, text):
        """Enhanced text encoding with multiple strategies"""
        if pd.isna(text) or text == "":
            text = ""
        
        # Choose strategy based on text length
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= self.max_text_len:
            # Short enough - use full text
            processed_text = text
        else:
            processed_text = self.dynamic_truncate(text)
        
        # Tokenize text
        inputs = self.tokenizer(
            processed_text, 
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
            # BiomedVLP-CXR-BERT uses last_hidden_state: (1, 512, 768)
            # Use [CLS] token (first token) from last hidden state
            text_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()  # Shape: (768,)
            
        return text_embedding

    def dynamic_truncate(self, text, max_len=512):
        """Randomly truncate text to preserve different parts"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_len - 2:  # -2 for [CLS] and [SEP]
            return text
        
        # Strategy 1: Random start position (50% of time)
        if random.random() < 0.5:
            start_idx = random.randint(0, len(tokens) - (max_len - 2))
            selected_tokens = tokens[start_idx:start_idx + (max_len - 2)]
        # Strategy 2: Take beginning + random middle/end (50% of time)
        else:
            # Always keep first 25% of tokens
            keep_start = len(tokens) // 4
            remaining_budget = max_len - 2 - keep_start
            
            if remaining_budget > 0:
                # Random sample from the rest
                remaining_tokens = tokens[keep_start:]
                if len(remaining_tokens) > remaining_budget:
                    selected_end = random.sample(remaining_tokens, remaining_budget)
                else:
                    selected_end = remaining_tokens
                selected_tokens = tokens[:keep_start] + selected_end
            else:
                selected_tokens = tokens[:max_len-2]
        
        return self.tokenizer.decode(selected_tokens)