import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import Unimodal_Based_Model, Multimodal_Based_Model, CSCQueue
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def single_modal_csc_loss(emb: torch.Tensor,
                           logits: torch.Tensor,
                           true_labels: torch.Tensor,
                           queue: CSCQueue,
                           tau: float = 0.1) -> torch.Tensor:
    """
    Compute Confidence-aware Supervised Contrastive Loss for single modality.

    emb: normalized features (batch_size, dim)
    logits: model outputs before softmax (batch_size, num_classes)
    true_labels: ground-truth labels (batch_size,)
    queue: CSCQueue object
    tau: temperature
    """
    # Normalize embeddings
    z = F.normalize(emb, dim=1)  # (B, D)
    preds = logits.argmax(dim=1)
    sr, _ = logits.softmax(dim=1).max(dim=1)  # confidence scores (B,)

    losses = []
    eps = 1e-8  # 防止log(0)

    for i in range(z.size(0)):
        anchor, y, c, w = z[i], true_labels[i], preds[i], sr[i]
        # get positive and negative samples for class y
        P, N = queue.get_positive_negative(y)
        if P.numel() == 0 or N.numel() == 0:
            continue
        P = F.normalize(P, dim=1)  # 正样本归一化
        N = F.normalize(N, dim=1)  # 负样本归一化
        # compute similarities
        pos_sims = torch.matmul(P, anchor) / tau  # (num_pos,)
        neg_sims = torch.matmul(N, anchor) / tau  # (num_neg,)

        all_sims = torch.cat([pos_sims, neg_sims], dim=0)  # (num_pos + num_neg)
        # Numerator: exp(similarity of positives)
        numerator = pos_sims.exp().sum()
        # Denominator: sum over exp(similarity of all samples)
        denominator = all_sims.exp().sum()
        loss_i = - w * torch.log(numerator / (denominator + eps))  # 防止除0
        losses.append(loss_i)
    if len(losses) == 0:
        return torch.tensor(0., device=emb.device)
    return torch.stack(losses).mean()


def multi_modal_csc_loss(emb_list: list,
                          logit_list: list,
                          true_labels: torch.Tensor,
                          queue_list: list,
                          tau: float = 0.1) -> torch.Tensor:
    """
    Final stable CSC Loss across multiple modalities.
    emb_list: [emb_mod1, emb_mod2, ...], normalized
    logit_list: [logits_mod1, logits_mod2, ...]
    queue_list: [queue_mod1, queue_mod2, ...]
    """
    total_loss = 0.0
    eps = 1e-8

    # 单模态 CSC loss
    for emb, logits, queue in zip(emb_list, logit_list, queue_list):
        total_loss += single_modal_csc_loss(emb, logits, true_labels, queue, tau)

    # 跨模态对比
    for i in range(len(emb_list) - 1):
        for j in range(i + 1, len(emb_list)):
            zi = F.normalize(emb_list[i], dim=1)
            zj = F.normalize(emb_list[j], dim=1)

            preds_i = logit_list[i].argmax(dim=1)
            preds_j = logit_list[j].argmax(dim=1)
            sr_i, _ = logit_list[i].softmax(dim=1).max(dim=1)
            sr_j, _ = logit_list[j].softmax(dim=1).max(dim=1)

            weight = (sr_i + sr_j) / 2.0  # 平均置信度权重

            sim = (zi * zj).sum(dim=1)  # cos相似度 (batch,)

            # 正例：两个模态都正确
            correct_mask = (preds_i == true_labels) & (preds_j == true_labels)
            pos_sim = sim[correct_mask]
            neg_sim = sim[~correct_mask]
            weight_pos = weight[correct_mask]

            if pos_sim.numel() == 0 or neg_sim.numel() == 0:
                continue

            pos_exp = (pos_sim / tau).exp()
            neg_exp = (neg_sim / tau).exp()

            numerator = pos_exp.sum()
            denominator = torch.cat([pos_exp, neg_exp], dim=0).sum()

            if torch.isinf(numerator) or torch.isinf(denominator) or numerator.item() == 0:
                continue

            loss_cm = - (weight_pos.mean()) * torch.log((numerator + eps) / (denominator + eps))

            total_loss += loss_cm

    # Normalize by number of terms
    return total_loss / (len(emb_list) + 1)


def train_or_eval_model(args, model, dataloader, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer != None
    loss_func = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]


        final_logits, text_logit, audio_logit, video_logit, final_featrures, text_features, audio_features, video_features = model(textf, visuf, acouf, umask, qmask, lengths)

        umask_bool = umask.bool()
        labels_ = label[umask_bool]
        final_featrures = final_featrures[umask_bool]
        final_logits = final_logits[umask_bool]
        text_features = text_features[umask_bool]
        audio_features = audio_features[umask_bool]
        video_features = video_features[umask_bool]
        text_logit = text_logit[umask_bool]
        audio_logit = audio_logit[umask_bool]
        video_logit = video_logit[umask_bool]


        loss = loss_func(final_logits, labels_) + loss_func(text_logit, labels_) + loss_func(audio_logit, labels_) + loss_func(video_logit, labels_)

        pred_ = torch.argmax(final_logits, 1)
        preds.extend(pred_.data.cpu().numpy().tolist())
        labels.extend(labels_.data.cpu().numpy().tolist())
        masks.extend(umask.view(-1).cpu().numpy().tolist())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    avg_loss = round(np.sum(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--train', type=bool, default=True, help="Is or Isn't train")
    parser.add_argument('--modal', type=str, default='video', choices=['text', 'video', 'audio'], help='modal to train')
    args = parser.parse_args()

    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    D_text, D_video, D_audio = None, None, None
    n_classes, n_speakers = None, None
    train_loader, valid_loader, test_loader = None, None, None

    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                   batch_size=batch_size,
                                                                   num_workers=0)
        n_speakers = 9
        n_classes = 7
        D_audio = 300
        D_video = 342
        D_text = 1024

    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
        n_speakers = 2
        n_classes = 6
        D_audio = 1582
        D_video = 342
        D_text = 1024

    else:
        print("There is no such dataset")

    D_m = D_audio + D_video + D_text

    model = Multimodal_Based_Model(args.Dataset, args.temp, D_text, D_video, D_audio, args.n_head,
                                   n_classes=n_classes,
                                   hidden_dim=args.hidden_dim,
                                   n_speakers=n_speakers,
                                   dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    if args.train:
        for e in range(n_epochs):
            start_time = time.time()

            train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(args, model, train_loader, optimizer, True)
            valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(args, model, valid_loader, None, False)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(args, model, test_loader)
            all_fscore.append(test_fscore)

            if best_fscore == None or best_fscore < test_fscore:
                best_fscore = test_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                torch.save(model.state_dict(), f'./save_model/best_model_{args.Dataset}.pt')

            print(
                'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                       test_fscore, round(time.time() - start_time, 2)))
            if (e + 1) % 10 == 0:
                print(classification_report(best_label, best_pred, digits=4))
                print(confusion_matrix(best_label, best_pred))
    else:
        model.load_state_dict(torch.load(f'./save_model/best_model_{args.Dataset}_{args.modal}.pt'))
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(args, model, test_loader)
        print(classification_report(test_label, test_pred, digits=4))
        print(confusion_matrix(test_label, test_pred))






