import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
from typing import Tuple
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from multiobjective_optimization import *


def calculate_log_probs(
    outputs_pI,
    outputs_spec,
    good_labels_pI,
    bad_labels_pI,
    good_labels_spec,
    bad_labels_spec,
    ref_outputs_pI,
    ref_outputs_spec,
):
    # Compute log-probabilities for the model outputs
    log_probs_pI = F.log_softmax(outputs_pI.logits, dim=-1)
    log_probs_spec = F.log_softmax(outputs_spec.logits, dim=-1)

    # Process good labels for pI
    good_labels_without_negatives_pI = torch.where(
        good_labels_pI != -100, good_labels_pI, 0
    )
    good_log_probs_pI = log_probs_pI.gather(
        dim=-1, index=good_labels_without_negatives_pI.unsqueeze(-1)
    ).squeeze(-1)
    good_log_probs_pI = torch.where(good_labels_pI != -100, good_log_probs_pI, 0).sum(
        -1
    )

    # Process good labels for specificity
    good_labels_without_negatives_spec = torch.where(
        good_labels_spec != -100, good_labels_spec, 0
    )
    good_log_probs_spec = log_probs_spec.gather(
        dim=-1, index=good_labels_without_negatives_spec.unsqueeze(-1)
    ).squeeze(-1)
    good_log_probs_spec = torch.where(
        good_labels_spec != -100, good_log_probs_spec, 0
    ).sum(-1)

    # Process bad labels for pI
    bad_labels_without_negatives_pI = torch.where(
        bad_labels_pI != -100, bad_labels_pI, 0
    )
    bad_log_probs_pI = log_probs_pI.gather(
        dim=-1, index=bad_labels_without_negatives_pI.unsqueeze(-1)
    ).squeeze(-1)
    bad_log_probs_pI = torch.where(bad_labels_pI != -100, bad_log_probs_pI, 0).sum(-1)

    # Process bad labels for spec
    bad_labels_without_negatives_spec = torch.where(
        bad_labels_spec != -100, bad_labels_spec, 0
    )
    bad_log_probs_spec = log_probs_spec.gather(
        dim=-1, index=bad_labels_without_negatives_spec.unsqueeze(-1)
    ).squeeze(-1)
    bad_log_probs_spec = torch.where(
        bad_labels_spec != -100, bad_log_probs_spec, 0
    ).sum(-1)

    # Compute log-probabilities for the reference model outputs
    ref_log_probs_pI = F.log_softmax(ref_outputs_pI.logits, dim=-1)
    ref_log_probs_spec = F.log_softmax(ref_outputs_spec.logits, dim=-1)

    # Process reference good labels for pI
    ref_good_log_probs_pI = ref_log_probs_pI.gather(
        dim=-1, index=good_labels_without_negatives_pI.unsqueeze(-1)
    ).squeeze(-1)
    ref_good_log_probs_pI = torch.where(
        good_labels_pI != -100, ref_good_log_probs_pI, 0
    ).sum(-1)

    # Process reference good labels for specificity
    ref_good_log_probs_spec = ref_log_probs_spec.gather(
        dim=-1, index=good_labels_without_negatives_spec.unsqueeze(-1)
    ).squeeze(-1)
    ref_good_log_probs_spec = torch.where(
        good_labels_spec != -100, ref_good_log_probs_spec, 0
    ).sum(-1)

    # Process reference bad labels for pI
    ref_bad_log_probs_pI = ref_log_probs_pI.gather(
        dim=-1, index=bad_labels_without_negatives_pI.unsqueeze(-1)
    ).squeeze(-1)
    ref_bad_log_probs_pI = torch.where(
        bad_labels_pI != -100, ref_bad_log_probs_pI, 0
    ).sum(-1)

    # Process reference bad labels for pI
    ref_bad_log_probs_spec = ref_log_probs_spec.gather(
        dim=-1, index=bad_labels_without_negatives_spec.unsqueeze(-1)
    ).squeeze(-1)
    ref_bad_log_probs_spec = torch.where(
        bad_labels_spec != -100, ref_bad_log_probs_spec, 0
    ).sum(-1)

    return (
        good_log_probs_pI,
        bad_log_probs_pI,
        ref_good_log_probs_pI,
        ref_bad_log_probs_pI,
        good_log_probs_spec,
        bad_log_probs_spec,
        ref_good_log_probs_spec,
        ref_bad_log_probs_spec,
    )


def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (
            logits - 1 / (2 * beta)
        ) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def get_gradients(model, loss_fns):
    gradients = []
    for loss_fn in loss_fns:
        # Compute the gradients for the current loss
        grads = torch.autograd.grad(
            loss_fn, model.parameters(), allow_unused=True, create_graph=True
        )

        # Replace None gradients with zeros
        grads = [
            grad.detach() if grad is not None else torch.zeros_like(param)
            for grad, param in zip(grads, model.parameters())
        ]

        gradients.append(grads)

        # Clear the computational graph to free memory
        del grads
        torch.cuda.empty_cache()

    return gradients


def train_dpo(
    model,
    ref_model,
    train_dataloader,
    device,
    beta,
    num_epochs,
    lr,
):
    model.train()
    ref_model.eval()
    optimizer = AdamW(model.parameters(), lr=lr)
    best_loss = float("inf")
    tmp_loss = 1e9
    count = 0

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="./logs")

    # Set up Matplotlib for real-time plotting
    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots()
    # losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for iteration, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            (
                input_ids_pI,
                attention_mask_pI,
                input_ids_spec,
                attention_mask_spec,
                good_labels_pI,
                bad_labels_pI,
                good_labels_spec,
                bad_labels_spec,
            ) = batch

            input_ids_pI = input_ids_pI.to(device)
            attention_mask_pI = attention_mask_pI.to(device)
            input_ids_spec = input_ids_spec.to(device)
            attention_mask_spec = attention_mask_spec.to(device)

            good_labels_pI = good_labels_pI.to(device)
            good_labels_spec = good_labels_spec.to(device)

            bad_labels_pI = bad_labels_pI.to(device)
            bad_labels_spec = bad_labels_spec.to(device)

            with torch.no_grad():
                ref_outputs_pI = ref_model(
                    input_ids=input_ids_pI, attention_mask=attention_mask_pI
                )
                ref_outputs_spec = ref_model(
                    input_ids=input_ids_spec, attention_mask=attention_mask_spec
                )

            outputs_pI = model(input_ids=input_ids_pI, attention_mask=attention_mask_pI)
            outputs_spec = model(
                input_ids=input_ids_spec, attention_mask=attention_mask_spec
            )

            # Calculate log-probabilities
            (
                good_log_probs_pI,
                bad_log_probs_pI,
                ref_good_log_probs_pI,
                ref_bad_log_probs_pI,
                good_log_probs_spec,
                bad_log_probs_spec,
                ref_good_log_probs_spec,
                ref_bad_log_probs_spec,
            ) = calculate_log_probs(
                outputs_pI,
                outputs_spec,
                good_labels_pI,
                bad_labels_pI,
                good_labels_spec,
                bad_labels_spec,
                ref_outputs_pI,
                ref_outputs_spec,
            )

            dpo_loss_pI, chosen_rewards_pI, rejected_rewards_pI = preference_loss(
                good_log_probs_pI,
                bad_log_probs_pI,
                ref_good_log_probs_pI,
                ref_bad_log_probs_pI,
                beta,
                label_smoothing=0.0,
                ipo=False,
                reference_free=False,
            )

            dpo_loss_spec, chosen_rewards_spec, rejected_rewards_spec = preference_loss(
                good_log_probs_spec,
                bad_log_probs_spec,
                ref_good_log_probs_spec,
                ref_bad_log_probs_spec,
                beta,
                label_smoothing=0.0,
                ipo=False,
                reference_free=False,
            )

            torch.cuda.empty_cache()
            loss_fns = [dpo_loss_pI.mean(), dpo_loss_spec.mean()]

            # Get gradients for pI and specificity
            grads = get_gradients(model, loss_fns)

            torch.cuda.empty_cache()

            # Frank-Wolfe iteration to compute scales
            with torch.no_grad():
                scale, _ = MinNormSolver.find_min_norm_element(grads)
                print(scale)

            dpo_loss = scale[0] * dpo_loss_pI + scale[1] * dpo_loss_spec

            # First backward pass to compute gradients
            optimizer.zero_grad()

            dpo_loss = dpo_loss.mean()
            dpo_loss.backward()

            # Update model parameters
            optimizer.step()

            # optimizer.zero_grad()
            total_loss += dpo_loss.item()

            torch.cuda.empty_cache()

            # Log the loss to TensorBoard after each batch
            writer.add_scalar(
                "Loss/train", dpo_loss.item(), epoch * len(train_dataloader) + iteration
            )

            # Log weights and biases
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(
                        f"{name}/weights",
                        param.data.cpu().numpy(),
                        epoch * len(train_dataloader) + iteration,
                    )
                    writer.add_histogram(
                        f"{name}/gradients",
                        param.grad.cpu().numpy(),
                        epoch * len(train_dataloader) + iteration,
                    )

            # print(dpo_loss.item())

            if dpo_loss.item() < tmp_loss:
                tmp_loss = dpo_loss.item()
                torch.save(model, "best_model.pth")
                print("Model saved!")
                count = 0
            else:
                count = count + 1
            if count == 100:
                break

            # Append loss to list for plotting
            # losses.append(dpo_loss.item())

            # # Update plot
            # ax.clear()
            # ax.plot(losses, label="Loss")
            # ax.set_xlabel("Iteration")
            # ax.set_ylabel("Loss")
            # ax.legend()
            # plt.draw()
            # plt.pause(0.001)

    writer.close()
    # plt.ioff()
    # plt.show()
