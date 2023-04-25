from tqdm import tqdm

import torch
import torch.nn.functional as F

def inference_language_modeling(model, eval_dataloader, device):
    model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
        ending_shape = batch["ending_input_ids"].shape 
        # flatten
        header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
        ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
        
        # adding this line of code takes me more than an hour.
        # without adding torch.no_grad, GPU usage will muiltply by 4.
        with torch.no_grad():
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
        pbar.set_description(f"Total Accuracy: {total_accuracy:.4f}, Batch Accuracy: {batch_accuracy:.4f}")
    return total_accuracy

def inference_contrastive_decoding(amateur_model, expert_model, eval_dataloader, device):
    amateur_model.eval()
    expert_model.eval()
    predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
        ending_shape = batch["ending_input_ids"].shape 
        # flatten
        header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
        ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
        
        # key step: compute logits.
        with torch.no_grad():
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
        pbar.set_description(f"Total Accuracy: {total_accuracy:.4f}, Batch Accuracy: {batch_accuracy:.4f}")
    return total_accuracy

def compute_seq2seq_conditional_score(batch, model, device):
    # returns log_prob of p(y|x) for each batch
    
    # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
    ending_shape = batch["ending_input_ids"].shape 
    # flatten. both input_ids has 0 as padding token.
    header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
    header_attention_mask = batch["header_attention_mask"].view(-1, batch["header_attention_mask"].shape[-1]).to(device)
    ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(input_ids = header_input_ids, 
                        attention_mask = header_attention_mask,
                        labels = ending_input_ids)
    
    _, logits = outputs.loss, outputs.logits
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 0
    ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=0).detach().cpu()
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    log_prob = ce_loss.view(ending_shape).sum(dim=-1)
    return log_prob

def compute_causal_conditional_score(batch, model, device):
    # returns log_prob of p(y|x) for each batch
    padding_token = 50256
    
    # e.g., (batch_size, #option, ending_seq_len): (32, 2, 18)
    # ending_shape = batch["ending_input_ids"].shape 
    # flatten. both input_ids has 0 as padding token.
    header_input_ids = batch["header_input_ids"].view(-1, batch["header_input_ids"].shape[-1]).to(device)
    header_attention_mask = batch["header_attention_mask"].view(-1, batch["header_attention_mask"].shape[-1]).to(device)
    ending_input_ids = batch["ending_input_ids"].view(-1, batch["ending_input_ids"].shape[-1]).to(device)
    ending_attention_mask = batch["ending_attention_mask"].view(-1, batch["ending_attention_mask"].shape[-1]).to(device)

    input_ids = torch.cat((header_input_ids, ending_input_ids), dim=-1)
    attention_mask = torch.cat((header_attention_mask, ending_attention_mask), dim=-1)
    padding_tensor = torch.full(header_input_ids.shape, padding_token).to(device)
    labels = torch.cat((padding_tensor, ending_input_ids), dim=-1)

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        # outputs = model(input_ids = header_input_ids, 
        #                 attention_mask = header_attention_mask,
        #                 labels = ending_input_ids)
        outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels)
    
    _, logits = outputs.loss, outputs.logits
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 50256
    # ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=0).detach().cpu()
    ce_loss = F.cross_entropy(logits, labels.view(-1), reduction="none", ignore_index=padding_token).detach().cpu()
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    # log_prob = ce_loss.view(labels.shape).sum(dim=-1)
    log_prob = ce_loss.view(batch["ending_input_ids"].shape[0], batch["ending_input_ids"].shape[1], -1).sum(dim=-1)
    return log_prob