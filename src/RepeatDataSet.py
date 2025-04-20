from torch.utils.data import Dataset

class RepeatDataSet(Dataset):
    def __init__(self, base_dataset, repeat=3):
        self.base_dataset = base_dataset;
        self.repeat = repeat;

    def __len__(self):
        return len(self.base_dataset) * self.repeat;

    def __getitem__(self, idx):
        return self.base_dataset[idx % len(self.base_dataset)];
