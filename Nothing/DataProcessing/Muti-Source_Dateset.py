from torch.utils.data import ConcatDataset

# 假设 dataset1 和 dataset2 是两个 Dataset 对象
combined_dataset = ConcatDataset([dataset1, dataset2])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)