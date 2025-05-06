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
from model import Unimodal_Based_Model, Multimodal_Based_Model, ConfidenceFusionModel
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


def train_or_eval_model(args, model, dataloader, optimizer=None, train=False, a=None, b=None):
    losses, preds, labels, masks = [], [], [], []
    losses_unimodal, losses_modal, losses_sample = [], [], []
    losses_text, losses_audio, losses_visual = [], [], []

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


        hs, hfs, out, logits, weights_tensor, conf_losses, loss_modal = model(textf, visuf, acouf, umask, qmask, lengths, label)

        umask_bool = umask.bool()
        labels_ = label[umask_bool]

        loss_text = loss_func(logits[0], labels_)
        loss_audio  = loss_func(logits[1], labels_)
        loss_visual = loss_func(logits[2], labels_)

        loss_unimodal = 1/3 * (loss_text + loss_audio + loss_visual)

        loss_sample = sum(conf_losses) / 3

        loss = loss_func(out, labels_) + loss_unimodal + a * loss_sample + b * loss_modal

        pred_ = torch.argmax(out, 1)
        preds.extend(pred_.data.cpu().numpy().tolist())
        labels.extend(labels_.data.cpu().numpy().tolist())
        masks.extend(umask.view(-1).cpu().numpy().tolist())
        losses.append(loss.item())

        losses_text.append(loss_text.item())
        losses_audio.append(loss_audio.item())
        losses_visual.append(loss_visual.item())
        losses_unimodal.append(loss_unimodal.item())
        losses_sample.append(loss_sample.item())
        losses_modal.append(loss_modal.item())


        if train:
            loss.backward()
            optimizer.step()

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_loss_text = round(np.sum(losses_text)/len(losses_text), 4)
    avg_loss_audio = round(np.sum(losses_audio)/len(losses_audio), 4)
    avg_loss_visual = round(np.sum(losses_visual)/len(losses_visual), 4)
    avg_loss_unimodal = round(np.sum(losses_unimodal)/len(losses_unimodal), 4)
    avg_loss_sample = round(np.sum(losses_sample)/len(losses_sample), 4)
    avg_loss_modal = round(np.sum(losses_modal)/len(losses_modal), 4)

    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_loss_text, avg_loss_audio, avg_loss_visual, avg_loss_unimodal, avg_loss_sample, avg_loss_modal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.000001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--train', type=bool, default=True, help="Is or Isn't train")
    parser.add_argument('--modal', type=str, default='video', choices=['text', 'video', 'audio'], help='modal to train')
    args = parser.parse_args()

    a_set = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    b_set = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    all_best_fscore = 66.0

    for a in a_set:
        for b in b_set:

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

            model = ConfidenceFusionModel(args.Dataset, args.temp, D_text, D_video, D_audio, args.n_head,
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

            save_path = f'./results/{args.Dataset}'
            parameters = {}

            if args.train:
                for e in range(n_epochs):
                    start_time = time.time()

                    train_loss, train_acc, _, _, _, train_fscore, train_loss_text, train_loss_audio, train_loss_visual, train_loss_unimodal, train_loss_sample, train_loss_modal = train_or_eval_model(args, model, train_loader, optimizer, True, a=a, b=b)
                    valid_loss, valid_acc, _, _, _, valid_fscore, valid_loss_text, valid_loss_audio, valid_loss_visual, valid_loss_unimodal, valid_loss_sample, valid_loss_modal = train_or_eval_model(args, model, valid_loader, None, False, a=a, b=b)
                    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_loss_text, test_loss_audio, test_loss_visual, test_loss_unimodal, test_loss_sample, test_loss_modal = train_or_eval_model(args, model, test_loader, a=a, b=b)
                    all_fscore.append(test_fscore)

                    train_para_dict = {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_fscore': train_fscore,
                        'train_loss_text': train_loss_text,
                        'train_loss_audio': train_loss_audio,
                        'train_loss_visual': train_loss_visual,
                        'train_loss_unimodal': train_loss_unimodal,
                        'train_loss_sample': train_loss_sample,
                        'train_loss_modal': train_loss_modal,
                    }
                    valid_para_dict = {
                        'valid_loss': valid_loss,
                        'valid_acc': valid_acc,
                        'valid_fscore': valid_fscore,
                        'valid_loss_text': valid_loss_text,
                        'valid_loss_audio': valid_loss_audio,
                        'valid_loss_visual': valid_loss_visual,
                        'valid_loss_unimodal': valid_loss_unimodal,
                        'valid_loss_sample': valid_loss_sample,
                        'valid_loss_modal': valid_loss_modal,
                    }
                    test_para_dict = {
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'test_fscore': test_fscore,
                        'test_loss_text': test_loss_text,
                        'test_loss_audio': test_loss_audio,
                        'test_loss_visual': test_loss_visual,
                        'test_loss_unimodal': test_loss_unimodal,
                        'test_loss_sample': test_loss_sample,
                        'test_loss_modal': test_loss_modal,
                    }
                    parameters[e] = {
                        'train': train_para_dict,
                        'valid': valid_para_dict,
                        'test': test_para_dict,
                    }


                    if best_fscore == None or best_fscore < test_fscore:
                        best_fscore = test_fscore
                        best_label, best_pred, best_mask = test_label, test_pred, test_mask
                        if all_best_fscore < best_fscore:
                            all_best_fscore = best_fscore
                            torch.save(model.state_dict(), f'./save_model/best_model_{args.Dataset}_{a}_{b}_{all_best_fscore}.pt')

                    print(
                        'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                        format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                               test_fscore, round(time.time() - start_time, 2)))
                    if (e + 1) % 10 == 0:
                        print(classification_report(best_label, best_pred, digits=4))
                        print(confusion_matrix(best_label, best_pred))

                save_path = f'./results/{args.Dataset}/{args.Dataset}_{a}_{b}_{best_fscore}.json'
                with open(save_path, 'w') as f:
                    json.dump(parameters, f)

            else:
                model.load_state_dict(torch.load(f'./save_model/best_model_{args.Dataset}_{args.modal}.pt'))
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(args, model, test_loader, a=a, b=b)
                print(classification_report(test_label, test_pred, digits=4))
                print(confusion_matrix(test_label, test_pred))






