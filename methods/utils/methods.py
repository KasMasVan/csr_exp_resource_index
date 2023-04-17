from tqdm import tqdm

import torch
import torch.nn.functional as F

def inference_language_modeling(model, eval_dataloader, device):
    model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)

    with tqdm(eval_dataloader, desc="Inference", postfix=[{"Batch Accuracy": 0.0, "Total Accuracy": 0.0,}]) as t:
        for batch in t:
            # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
            ending_shape = batch["ending_input_ids"].shape 
            # flatten
            header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
            ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
            outputs = model(input_ids = header_input_ids, labels = ending_input_ids)
            _, logits = outputs.loss, outputs.logits
            # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
            logits = logits.view(-1, logits.shape[-1])
            # ignore padding token: 0
            ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=0).detach().cpu()
            # each score is the negative log-likelihood of a ending given a header.
            batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
            batch_labels = batch["label"]
            predictions = torch.cat((predictions, batch_predictions))
            labels = torch.cat((labels, batch_labels))
        
            # make accuracy accumulative
            batch_accuracy = (batch_predictions == batch_labels).sum().item() / len(batch_labels)
            total_accuracy = (predictions == labels).sum().item() / len(labels)
            t.postfix[-1]["Batch Accuracy"] = f"{batch_accuracy:.4f}"
            t.postfix[-1]["Total Accuracy"] = f"{total_accuracy:.4f}"
            t.update()
    return total_accuracy

def inference_contrastive_decoding(amateur_model, expert_model, eval_dataloader, device):
    amateur_model.eval()
    expert_model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)

    with tqdm(eval_dataloader, desc="Inference", postfix=[{"Batch Accuracy": 0.0, "Total Accuracy": 0.0,}]) as t:
        for batch in t:
            # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
            ending_shape = batch["ending_input_ids"].shape 
            # flatten
            header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
            ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
            # key step: compute logits.
            amateur_model_logits = amateur_model(input_ids = header_input_ids, labels = ending_input_ids).logits
            expert_model_logits = expert_model(input_ids = header_input_ids, labels = ending_input_ids).logits
            logits = expert_model_logits - amateur_model_logits
            # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
            logits = logits.view(-1, logits.shape[-1])
            # ignore padding token: 0
            ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=0).detach().cpu()
            # each score is the negative log-likelihood of a ending given a header.
            batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
            batch_labels = batch["label"]
            predictions = torch.cat((predictions, batch_predictions))
            labels = torch.cat((labels, batch_labels))
        
            # make accuracy accumulative
            batch_accuracy = (batch_predictions == batch_labels).sum().item() / len(batch_labels)
            total_accuracy = (predictions == labels).sum().item() / len(labels)
            t.postfix[-1]["Batch Accuracy"] = f"{batch_accuracy:.4f}"
            t.postfix[-1]["Total Accuracy"] = f"{total_accuracy:.4f}"
            t.update()
    return total_accuracy