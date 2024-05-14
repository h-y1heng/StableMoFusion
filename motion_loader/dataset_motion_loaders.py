from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def get_dataset_loader(opt, batch_size, mode='eval',split='test', accelerator=None):
    dataset = get_dataset(opt, split, mode, accelerator)
    if mode in ['eval','gt_eval']:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=True, collate_fn=collate_fn
        ) 
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, drop_last=True,persistent_workers=True
        )
    return dataloader