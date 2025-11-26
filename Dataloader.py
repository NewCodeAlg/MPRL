from typing import Tuple
from torch.utils.data import DataLoader
import h5py
import torch
from torch.utils.data import Dataset
import re
from torch.nn.utils.rnn import pad_sequence

def SimpleColl(batch):
    embeddings = [item[0] for item in batch]
    solubility = torch.tensor([item[1] for item in batch])
    len = torch.tensor([item[2] for item in batch])
    embeddings = pad_sequence(embeddings, batch_first=True)
    
    return embeddings, solubility,len

class SimpleDataSet(Dataset):
    def __init__(self, embeddings_path,
                 fasta_path,
                 transform=lambda x: x) -> None:
        super().__init__()
        self.transform = transform
        self.embeddings_file = h5py.File(embeddings_path, 'r')
        self.sequencesid = list()
        self.sequencesdict = dict()
        self.lenNameDict = dict()
        self.MaxLength = 1024
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                    self.sequencesid.append(uniprot_id)
                    self.sequencesdict[uniprot_id] = ''
                else:
                    self.sequencesdict[uniprot_id] += ''.join(line.split()).upper().replace("-","")  
        for id in self.sequencesid:
            self.lenNameDict[id] = len(self.sequencesdict[id])
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences_id = self.sequencesid[index]
        label_match = re.search(r'\bA-(\d+)\b', sequences_id)
        if not label_match:
            raise ValueError(f"Invalid label format in {sequences_id}")
        if self.lenNameDict[sequences_id] > self.MaxLength:
            length = self.MaxLength
            label = int(label_match.group(1))
            embedding = self.embeddings_file[sequences_id][:self.MaxLength,:]
            embedding, solubility,length = self.transform((embedding, label,length))
        else :
            label = int(label_match.group(1))
            length = self.lenNameDict[sequences_id]
            embedding = self.embeddings_file[sequences_id][:]
            embedding, solubility,length = self.transform((embedding, label,length))
        return embedding, solubility,length
    def __len__(self) -> int:
        return len(self.sequencesid)

    def LoaderReturnMethod(self,batchSize = 8 ,coll = SimpleColl)-> DataLoader:
        return   DataLoader(dataset=self,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=coll
    )




