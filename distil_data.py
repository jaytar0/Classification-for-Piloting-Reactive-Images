import torch

class distil_data(torch.utils.data.Dataset):

    def __init__(self, enc, lab):

        self.encodings = enc
        self.labels = lab

    def __getitem__(self, i):

        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][i]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][i]),
            'labels': torch.tensor(self.labels[i])
        }
    def __len__(self):
        return len(self.labels)
    