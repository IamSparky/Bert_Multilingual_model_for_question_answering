import config
import utils
import torch
from tqdm import tqdm
import numpy
import gc
import torch.nn as nn

def loss_fn(op1, op2, act1, act2):
    loss1 = nn.BCEWithLogitsLoss()(op1, act1)
    loss2 = nn.BCEWithLogitsLoss()(op2, act2)
    return loss1 + loss2

def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    tqdm_ob = tqdm(data_loader, total = len(data_loader))
    
    for bi, d in enumerate(tqdm_ob):

        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tqdm_ob.set_postfix(loss = losses.avg)

        del ids, mask, token_type_ids, targets_start, targets_end
        gc.collect()
        torch.cuda.empty_cache()


def eval_fn(data_loader, model, device):
    model.eval()

    tqdm_ob = tqdm(data_loader, total = len(data_loader))
    
    for bi, d in enumerate(tqdm_ob):

        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        context = d["context"]
        question = d["question"]
        answer = d["answer"]
        padding_len = d["padding_len"]
        

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            metric_score, _ = utils.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            jaccard_scores.append(metric_score)
    
        losses.update(loss.item(), ids.size(0))
        tqdm_ob.set_postfix(loss = losses.avg)

        del ids, mask, token_type_ids, targets_start, targets_end
        gc.collect()
        torch.cuda.empty_cache()

        return np.mean(jaccard_scores)