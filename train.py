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
from model import Unimodal_Based_Model
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

        modal_input = None
        if args.modal == 'text':
            modal_input = textf
        elif args.modal == 'video':
            modal_input = visuf
        elif args.modal == 'audio':
            modal_input = acouf

        modal_logits, modal_features = model(modal_input, umask, qmask, lengths)


        umask_bool = umask.bool()
        labels_ = label[umask_bool]
        modal_features = modal_features[umask_bool]
        modal_logits = modal_logits[umask_bool]

        loss = loss_func(modal_logits, labels_)


        pred_ = torch.argmax(modal_logits, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
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
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
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
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    D_text, D_video, D_audio = None, None, None
    n_classes, n_speakers = None, None

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

    model = None
    if args.modal == 'text':
        model = Unimodal_Based_Model(args.Dataset, args.temp, D_text, args.n_head,
                                    n_classes=n_classes,
                                    hidden_dim=args.hidden_dim,
                                    n_speakers=n_speakers,
                                    dropout=args.dropout)
    elif args.modal == 'audio':
        model = Unimodal_Based_Model(args.Dataset, args.temp, D_audio, args.n_head,
                                     n_classes=n_classes,
                                     hidden_dim=args.hidden_dim,
                                     n_speakers=n_speakers,
                                     dropout=args.dropout)
    elif args.modal == 'video':
        model = Unimodal_Based_Model(args.Dataset, args.temp, D_video, args.n_head,
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

            if best_fscore == None or best_fscore < valid_fscore:
                best_fscore = valid_fscore
                best_label, best_pred, best_mask = test_label, test_pred, test_mask
                torch.save(model.state_dict(), f'./save_model/best_model_{args.Dataset}_{args.modal}.pt')

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


    print(classification_report(best_label, best_pred, digits=4))
    print(confusion_matrix(best_label, best_pred))





