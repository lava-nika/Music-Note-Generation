import os
import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, root_dir, context_size=196, overlapping=True):
        
        self.context_size = context_size
        self.data = []
        
        num_files = 0

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.pt'):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}") # for debugging 
                    song_tensor = torch.load(file_path)  
                    if song_tensor.ndim != 2 or song_tensor.shape[1] != 4:
                        print(f"Skipping invalid tensor shape: {song_tensor.shape}")
                        continue
                    self._process_song(song_tensor, overlapping)
                    num_files += 1
        
        if len(self.data) == 0:
            raise ValueError("No valid data points were found. Check file paths/logic.")
    
        print(f"Processed {num_files} files, found {len(self.data)} valid data points.")
        


    def _process_song(self, song_tensor, overlapping):
        L = song_tensor.shape[0]
        step = 1 if overlapping else self.context_size
        
        for i in range(0, L - self.context_size, step):
            context = song_tensor[i:i + self.context_size]
            target = song_tensor[i + self.context_size]
            self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]