from tqdm import tqdm

import torch
import torch.nn.functional as F

def inference_language_modeling_old(model, eval_dataloader, device):
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

def inference_contrastive_decoding_old(amateur_model, expert_model, eval_dataloader, device):
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

def inference_language_modeling(model, eval_dataloader, device, compute_func, pad_token_id):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)
        
        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat((avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))
    
        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(labels)
        pbar.set_description(f"Language modeling accuracy: {lm_accuracy:.4f}, Average language modeling accuracy: {avg_lm_accuracy:.4f}")
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy

def inference_calibration(model, eval_dataloader, eval_calibration_dataloader, device, compute_func, pad_token_id):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()
    avg_log_probs = []

    pbar = tqdm(zip(eval_dataloader, eval_calibration_dataloader), desc="Inference", total=len(eval_dataloader))
    for batch, batch_calibration in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        log_prob_calibration = compute_func(batch_calibration, model, device, pad_token_id)
        log_prob = log_prob - log_prob_calibration
        avg_log_prob = log_prob / batch["ending_attention_mask"].sum(dim=-1)
        avg_log_probs.append(avg_log_prob)

        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = avg_log_prob.argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat((avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))
    
        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(labels)
        pbar.set_description(f"Calibration accuracy: {lm_accuracy:.4f}, Average calibration accuracy: {avg_lm_accuracy:.4f}")
    avg_log_probs = torch.cat(avg_log_probs, dim=0)
    return avg_log_probs, lm_accuracy, avg_lm_accuracy

def compute_mask_process_of_elimination(avg_log_probs):
    masks = torch.ones_like(avg_log_probs)
    # # soft masking (v1), i.e., get rid of the least likely answer.
    # masks[torch.arange(avg_log_probs.shape[0]), avg_log_probs.argmin(dim=-1)] = 0
    
    # v2: Calculate the row-wise mean
    row_mean = avg_log_probs.mean(dim=1, keepdim=True)
    # Set values below the mean to 0
    masks[avg_log_probs > row_mean] = 0

    return masks

def inference_process_of_elimination(model, eval_dataloader, device, compute_func, pad_token_id):
    model.eval()
    lm_predictions = torch.zeros(0)
    avg_lm_predictions = torch.zeros(0)
    labels = torch.zeros(0)
    torch.cuda.empty_cache()

    pbar = tqdm(eval_dataloader, desc="Inference")
    for batch in pbar:
        log_prob = compute_func(batch, model, device, pad_token_id)
        # apply hard masking
        log_prob[batch["mask"] == 0] = float("inf")
        
        ending_length = batch["ending_attention_mask"].sum(dim=-1)
        batch_predictions = log_prob.argmin(dim=-1)
        batch_avg_predictions = (log_prob / ending_length).argmin(dim=-1)

        batch_labels = batch["label"]
        lm_predictions = torch.cat((lm_predictions, batch_predictions))
        avg_lm_predictions = torch.cat((avg_lm_predictions, batch_avg_predictions))
        labels = torch.cat((labels, batch_labels))
    
        # make accuracy accumulative
        lm_accuracy = (lm_predictions == labels).sum().item() / len(labels)
        avg_lm_accuracy = (avg_lm_predictions == labels).sum().item() / len(labels)
        pbar.set_description(f"Process of elimination accuracy: {lm_accuracy:.4f}, Average process of elimination accuracy: {avg_lm_accuracy:.4f}")
    return lm_accuracy, avg_lm_accuracy

def compute_conditional_score_seq2seq(batch, model, device, pad_token_id):
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
    ce_loss = F.cross_entropy(logits, ending_input_ids.view(-1), reduction="none", ignore_index=pad_token_id).detach().cpu()
    # each score is the negative log-likelihood of a ending given a header.
    # batch_predictions = ce_loss.view(ending_shape).sum(dim=-1).argmin(dim=-1)
    log_prob = ce_loss.view(ending_shape).sum(dim=-1)
    return log_prob

def compute_conditional_score_causal(batch, model, device, pad_token_id):
    # returns log_prob of p(y|x) for each batch
    # make sure the padding token is aligned with tokenizer.pad_token_id 
    # and preprocess_function_causal
    # padding_token = 50256
    
    input_ids = batch["input_ids"].view(-1, batch["input_ids"].shape[-1]).to(device)
    labels = batch["labels"].view(-1, batch["labels"].shape[-1]).to(device)

    # adding this line of code takes me more than an hour.
    # without adding torch.no_grad, GPU usage will muiltply by 4.
    with torch.no_grad():
        outputs = model(input_ids = input_ids,
                        # attention_mask = attention_mask,
                        labels = labels)
    
    _, logits = outputs.loss, outputs.logits
    # shift
    logits = logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    # e.g., (batch_size * #option, ending_seq_len, #vocab): (64, 18, 32128)
    logits = logits.view(-1, logits.shape[-1])
    # ignore padding token: 50256
    ce_loss = F.cross_entropy(logits, labels.view(-1), reduction="none", ignore_index=pad_token_id).detach().cpu()
    # each score is the negative log-likelihood of a ending given a header.
    log_prob = ce_loss.view(batch["input_ids"].shape[0], batch["input_ids"].shape[1], -1).sum(dim=-1)
    return log_prob