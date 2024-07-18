import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import pandas as pd
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

tokenizer = torch.load("PepMLM-650M-tokenizer.pth")
model = torch.load("PepMLM-650M.pth")
ref_model = torch.load("PepMLM-650M.pth")


class ProteinDataset(Dataset):
    def __init__(self, file, tokenizer):
        data = pd.read_csv(file)
        self.tokenizer = tokenizer
        self.proteins = data["Receptor Sequence"].tolist()
        self.peptides = data["Binder"].tolist()
        self.decoy_binders = data["Decoy Binder"].tolist()

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        protein_seq = self.proteins[idx]
        peptide_seq = self.peptides[idx]
        decoy_peptide_seq = self.decoy_binders[idx]

        return protein_seq, peptide_seq, decoy_peptide_seq


def train_dpo(
    model,
    ref_model,
    tokenizer,
    train_dataset,
    test_dataset,
    device,
    beta,
    batch_size,
    num_epochs,
):
    loss_tmp = 1e9
    model.train()
    ref_model.eval()  # Set the reference model to evaluation mode
    optimizer = AdamW(model.parameters(), lr=lr / batch_size)

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(train_dataset), 1):
            tmp = train_dataset[i : i + 1]
            batch = list(zip(tmp[0], tmp[1], tmp[2]))
            input_ids_ = []
            attention_masks = []
            good_seq_labels = []
            bad_seq_labels = []

            for protein_seq, peptide_seq, decoy_peptide_seq in batch:

                masked_peptide = "<mask>" * len(peptide_seq)
                complex_seq = protein_seq + masked_peptide

                # Tokenize and pad the complex sequence
                complex_input = tokenizer(
                    complex_seq,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )

                input_ids = complex_input["input_ids"].squeeze()
                attention_mask = complex_input["attention_mask"].squeeze()

                # Create labels
                label_seq_good = protein_seq + peptide_seq
                labels_good = tokenizer(
                    label_seq_good,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )["input_ids"].squeeze()

                # Set non-masked positions in the labels tensor to -100
                labels_good = torch.where(
                    input_ids == tokenizer.mask_token_id, labels_good, -100
                )

                label_seq_bad = protein_seq + decoy_peptide_seq
                labels_bad = tokenizer(
                    label_seq_bad,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )["input_ids"].squeeze()

                # Set non-masked positions in the labels tensor to -100
                labels_bad = torch.where(
                    input_ids == tokenizer.mask_token_id, labels_bad, -100
                )

                input_ids_.append(input_ids)
                attention_masks.append(attention_mask)
                good_seq_labels.append(labels_good)
                bad_seq_labels.append(labels_bad)

            input_ids = torch.stack(input_ids_).to(device)
            attention_mask = torch.stack(attention_masks).to(device)
            good_labels = torch.stack(good_seq_labels).to(device)
            bad_labels = torch.stack(bad_seq_labels).to(device)

            with torch.no_grad():
                ref_outputs = ref_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # log-probabilities
            log_probs = F.log_softmax(outputs.logits, dim=-1)

            # get log probabilities of good and bad labels
            good_labels_without_negatives = torch.where(good_labels != -100, good_labels, 0)
            good_log_probs = log_probs.gather(dim=-1, index=good_labels_without_negatives.unsqueeze(-1)).squeeze(-1)
            good_log_probs = torch.where(good_labels != -100, good_log_probs, 0).sum(-1)

            bad_labels_without_negatives = torch.where(bad_labels != -100, bad_labels, 0)
            bad_log_probs = log_probs.gather(dim=-1, index=bad_labels_without_negatives.unsqueeze(-1)).squeeze(-1)
            bad_log_probs = torch.where(bad_labels != -100, bad_log_probs, 0).sum(-1)

            # same for reference model
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)

            ref_good_log_probs = ref_log_probs.gather(dim=-1, index=good_labels_without_negatives.unsqueeze(-1)).squeeze(-1)
            ref_good_log_probs = torch.where(good_labels != -100, ref_good_log_probs, 0).sum(-1)

            ref_bad_log_probs = ref_log_probs.gather(dim=-1, index=bad_labels_without_negatives.unsqueeze(-1)).squeeze(-1)
            ref_bad_log_probs = torch.where(bad_labels != -100, ref_bad_log_probs, 0).sum(-1)

            # log probability differences for DPO loss
            good_diff = beta * (good_log_probs - ref_good_log_probs)
            bad_diff = beta * (bad_log_probs - ref_bad_log_probs)

            # Calculate DPO loss
            dpo_loss = -F.logsigmoid(
                good_diff.mean(dim=-1) - bad_diff.mean(dim=-1)
            ).mean()

            dpo_loss.backward()
            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += dpo_loss.item()

                iteration = i // batch_size + 1
                if iteration % 1000 == 0:
                    print(f"Iteration {iteration}, Loss: {dpo_loss.item()}")

        avg_loss = (total_loss / len(train_dataset)) * batch_size
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}")

        # Validation step
        val_loss = validate_model(model, test_dataset, tokenizer, device)

        if val_loss < loss_tmp:
            loss_tmp = val_loss
            torch.save(model, "PepMLM-650M_DPO_best_model.pth")
            print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {val_loss}")
            print(f"Model saved at epoch {epoch+1}")


def validate_model(model, dataset, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(dataset), 1):
            tmp = dataset[i : i + batch_size]
            batch = list(zip(tmp[0], tmp[1], tmp[2]))

            input_ids_ = []
            attention_masks = []
            good_seq_labels = []
            bad_seq_labels = []

            for protein_seq, peptide_seq, decoy_peptide_seq in batch:
                
                masked_peptide = "<mask>" * len(peptide_seq)
                complex_seq = protein_seq + masked_peptide

                
                complex_input = tokenizer(
                    complex_seq,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )

                input_ids = complex_input["input_ids"].squeeze()
                attention_mask = complex_input["attention_mask"].squeeze()

                # Create labels
                label_seq_good = protein_seq + peptide_seq
                labels_good = tokenizer(
                    label_seq_good,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )["input_ids"].squeeze()

                # Set non-masked positions in the labels tensor to -100
                labels_good = torch.where(
                    input_ids == tokenizer.mask_token_id, labels_good, -100
                )

                label_seq_bad = protein_seq + decoy_peptide_seq
                labels_bad = tokenizer(
                    label_seq_bad,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=552,
                    truncation=True,
                )["input_ids"].squeeze()

                # Set non-masked positions in the labels tensor to -100
                labels_bad = torch.where(
                    input_ids == tokenizer.mask_token_id, labels_bad, -100
                )

                input_ids_.append(input_ids)
                attention_masks.append(attention_mask)
                good_seq_labels.append(labels_good)
                bad_seq_labels.append(labels_bad)

            input_ids = torch.stack(input_ids_).to(device)
            attention_mask = torch.stack(attention_masks).to(device)
            good_labels = torch.stack(good_seq_labels).to(device)
            bad_labels = torch.stack(bad_seq_labels).to(device)

            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # log-probabilities
            log_probs = F.log_softmax(outputs.logits, dim=-1)

            # get log probabilities of good  and labels
            good_labels_without_negatives = torch.where(
                good_labels != -100, good_labels, 0
            )
            good_log_probs = log_probs.gather(
                dim=-1, index=good_labels_without_negatives.unsqueeze(-1)
            ).squeeze(-1)
            good_log_probs = torch.where(good_labels != -100, good_log_probs, 0).sum(-1)

            bad_labels_without_negatives = torch.where(
                bad_labels != -100, bad_labels, 0
            )
            bad_log_probs = log_probs.gather(
                dim=-1, index=bad_labels_without_negatives.unsqueeze(-1)
            ).squeeze(-1)
            bad_log_probs = torch.where(bad_labels != -100, bad_log_probs, 0).sum(-1)

            # same for reference model
            ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)

            ref_good_log_probs = ref_log_probs.gather(
                dim=-1, index=good_labels_without_negatives.unsqueeze(-1)
            ).squeeze(-1)
            ref_good_log_probs = torch.where(
                good_labels != -100, ref_good_log_probs, 0
            ).sum(-1)

            ref_bad_log_probs = ref_log_probs.gather(
                dim=-1, index=bad_labels_without_negatives.unsqueeze(-1)
            ).squeeze(-1)
            ref_bad_log_probs = torch.where(
                bad_labels != -100, ref_bad_log_probs, 0
            ).sum(-1)

            # log probability differences for DPO loss
            good_diff = beta * (good_log_probs - ref_good_log_probs)
            bad_diff = beta * (bad_log_probs - ref_bad_log_probs)

            # Calculate DPO loss
            dpo_loss = -F.logsigmoid(
                good_diff.mean(dim=-1) - bad_diff.mean(dim=-1)
            ).mean()

            total_loss += dpo_loss.item()

    avg_loss = (total_loss / len(dataset)) * batch_size
    return avg_loss

    model.train()


# Parameters
beta = 0.1
lr = 0.0007984276816171436
batch_size = 2
num_epochs = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ProteinDataset("./scripts/train_only_pI.csv", tokenizer)
test_dataset = ProteinDataset("./scripts/test_only_pI.csv", tokenizer)

train_dpo(
    model.to(device),
    ref_model.to(device),
    tokenizer,
    train_dataset,
    test_dataset,
    device,
    beta,
    batch_size,
    num_epochs,
)
