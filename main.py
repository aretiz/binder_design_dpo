import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_mo import *


class ProteinDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_seq, peptide_seq, decoy_pI_seq, decoy_spec_seq = self.data[idx]

        # Input ids and mask for pI
        masked_peptide_pI = "<mask>" * len(peptide_seq)
        complex_seq_pI = protein_seq + masked_peptide_pI

        # Tokenize and pad the complex sequence
        complex_input_pI = self.tokenizer(
            complex_seq_pI,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids_pI = complex_input_pI["input_ids"].squeeze()
        attention_mask_pI = complex_input_pI["attention_mask"].squeeze()

        # Create good labels for pI
        label_seq_good_pI = protein_seq + peptide_seq
        labels_good_pI = self.tokenizer(
            label_seq_good_pI,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels_good_pI = torch.where(
            input_ids_pI == self.tokenizer.mask_token_id, labels_good_pI, -100
        )

        # Labels for non-prefered pI
        label_seq_bad_pI = protein_seq + decoy_pI_seq
        labels_bad_pI = self.tokenizer(
            label_seq_bad_pI,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels_bad_pI = torch.where(
            input_ids_pI == self.tokenizer.mask_token_id, labels_bad_pI, -100
        )

        # Input ids and mask for specificity
        masked_peptide_spec = "<mask>" * len(decoy_spec_seq)
        complex_seq_spec = protein_seq + masked_peptide_spec

        complex_input_spec = self.tokenizer(
            complex_seq_spec,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids_spec = complex_input_spec["input_ids"].squeeze()
        attention_mask_spec = complex_input_spec["attention_mask"].squeeze()

        # Create good labels for specificity
        labels_good_spec = labels_good_pI

        # Labels for non-prefered specificity
        label_seq_bad_spec = protein_seq + decoy_spec_seq
        labels_bad_spec = self.tokenizer(
            label_seq_bad_spec,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )["input_ids"].squeeze()

        # Set non-masked positions in the labels tensor to -100
        labels_bad_spec = torch.where(
            input_ids_spec == self.tokenizer.mask_token_id, labels_bad_spec, -100
        )

        return (
            input_ids_pI,
            attention_mask_pI,
            input_ids_spec,
            attention_mask_spec,
            labels_good_pI,
            labels_good_spec,
            labels_bad_pI,
            labels_bad_spec,
        )


tokenizer = torch.load("PepMLM-650M-tokenizer.pth")
model = torch.load("PepMLM-650M.pth")
ref_model = torch.load("PepMLM-650M.pth")

# Parameters
beta = 0.2
lr = 1e-5
batch_size = 2
num_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = "./scripts/train_few.csv"

df = pd.read_csv(train_data_path)
train_dataset = df[
    ["Receptor_Sequence", "Binder", "Decoy_Binder", "Non_Original_Binder"]
].values.tolist()

train_dataloader = DataLoader(
    ProteinDataset(train_dataset, tokenizer, max_length=552),
    batch_size=batch_size,
    shuffle=True,
)

train_dpo(
    model.to(device),
    ref_model.to(device),
    train_dataloader,
    device,
    beta,
    num_epochs,
    lr=lr,
)
