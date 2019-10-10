import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dense_coattn.config import get_train_config
from dense_coattn.data import default_collate, VQADataset, BatchSampler
from dense_coattn.evaluate import VQA, VQAEval, evaluate
from dense_coattn.model import DCN, DCNWithRCNN
from dense_coattn.modules import LargeEmbedding
from dense_coattn.optim import FixedAdam
from dense_coattn.optim.lr_scheduler import StepScheduler
from dense_coattn.utils import (AverageMeter, Initializer, StopwatchMeter,
                                TimeMeter, extract_statedict, move_to_cuda,
                                save_checkpoint)


from evaluate import evalEpoch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

def trainEpoch(epoch, dataloader, model, criterion, optimizer, scheduler, opt, writer):
    torch.set_grad_enabled(True)
    model.train()
    batch_time = StopwatchMeter()
    data_time = StopwatchMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    ups = TimeMeter()

    scheduler.step(epoch + 1)
    data_time.start()
    batch_time.start()
    for i, batch in enumerate(dataloader):
        data_time.stop()
        lr = scheduler.step_update(epoch * len(dataloader) + i + 1)
        optimizer.zero_grad()

        img_info, ques, ques_mask, ans_idx = move_to_cuda(batch[:-1], devices=opt.gpus)
        ques = model.word_embedded(ques).detach()
        if False:
            img, img_mask = img_info
        else:
            img = img_info
            img_mask = img_info

        score = model(img, ques, img_mask, ques_mask)
        loss = criterion(score, ans_idx)
        losses.update(loss.item())

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy = evaluate(score.detach(), ans_idx)
        accuracies.update(accuracy.item())

        if writer is not None:
            for group_id, group in enumerate(optimizer.get_stats):
                writer.add_scalar(f"statistics/update_ratio_{group_id}", float(group[0]) / group[1],
                                  global_step=(epoch * len(dataloader) + i))
                writer.add_scalar(f"statistics/update_{group_id}", group[0], global_step=(epoch * len(dataloader) + i))

            writer.add_scalar("iter/train_loss", losses.avg, global_step=(epoch * len(dataloader) + i))
            writer.add_scalar("iter/train_accuracy", accuracies.avg, global_step=(epoch * len(dataloader) + i))
            writer.add_scalar("iter/lr", lr, global_step=(epoch * len(dataloader) + i))

        ups.update()
        batch_time.stop()
        if (i + 1) % opt.log_interval == 0:
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                  'Ups: {ups.avg:.3f}\t'
                  'Accuracy: {accuracy.val:.4f} ({accuracy.avg:.4f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch + 1, i + 1, len(dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, ups=ups, accuracy=accuracies,
            ))
            batch_time.reset()
            data_time.reset()
            ups.reset()

        batch_time.start()
        data_time.start()

        if (epoch * len(dataloader) + i + 1) >= opt.num_iter and opt.num_iter > 0:
            break

    if writer is not None:
        writer.add_scalar("epoch/train_loss", losses.avg, global_step=epoch)
        writer.add_scalar("epoch/train_accuracy", accuracies.avg, global_step=epoch)

    return losses.avg, accuracies.avg

def trainModel(trainLoader, valLoader, model, criterion, optimizer, scheduler, checkpoint, idx2ans, opt):
    best_accuracy = 0.
    start_epoch = 0
    bad_counter = 0
    history = []

    writer = None
    if opt.use_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir="logs/%s" % opt.model)

    if checkpoint is not None:
        best_accuracy = checkpoint["best_accuracy"]
        start_epoch = checkpoint["last_epoch"]
        bad_counter = checkpoint["bad_counter"]
        history = checkpoint["history"]

    if True:
        for epoch in range(start_epoch, opt.num_epoch):
            Initializer.manual_seed(opt.seed + epoch)
            print("----------------------------------------------")
            train_loss, train_accuracy = trainEpoch(epoch, trainLoader, model, criterion, optimizer, scheduler, opt, writer)
            print(">>>> Train [{:.3f}] \t loss: {:.3f} \t accuracy: {:.3f}".format(epoch, train_loss, train_accuracy))

            is_best = False
            is_save = False
            if valLoader is not None:
                val_loss, val_accuracy = evalEpoch(epoch, valLoader, model, criterion, opt, writer)
                print(">>>> Val [{:.3f}] \t loss: {:.3f} \t accuracy: {:.3f}".format(epoch, val_loss, val_accuracy))
                history.append(val_accuracy)

                if best_accuracy <= val_accuracy:
                    best_accuracy = val_accuracy
                    bad_counter = 0
                    is_best = True

                if (len(history) > opt.patience) and val_accuracy <= torch.Tensor(history[:-opt.patience]).max():
                    bad_counter += 1
                    if bad_counter > opt.patience:
                        print("** Early Stop!")
                        break

            if (epoch + 1) % opt.save_freq == 0:
                is_save = True

            is_parallel = True if len(opt.gpus) > 1 else False
            model_state_dict = extract_statedict(model, excludes=["word_embedded"], is_parallel=is_parallel)
            checkpoint = {
                "last_epoch": epoch + 1,
                "args": opt,
                "state_dict": model_state_dict,
                "best_accuracy": best_accuracy,
                "bad_counter": bad_counter,
                "history": history,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
            }
            save_checkpoint(opt.model, checkpoint, is_best, is_save, opt.directory)

    if False:
        try:
            if valLoader is not None:
                vqaEval(valLoader, model, criterion, idx2ans, opt)
        except Exception as e:
            print(">>>> Exception:", e)

    if writer is not None:
        writer.close()

def main(opt):
    print(">> Creating saving folder if it does not exist: {}".format(opt.directory))
    if not os.path.exists(opt.directory):
        os.makedirs(opt.directory)

    checkpoint = None
    if opt.resume:
        opt.resume = os.path.join(opt.directory, opt.resume)
        if os.path.isfile(opt.resume):
            print(">>>> Loading checkpoint {}".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            if opt.overwrite:
                opt = checkpoint["args"]
                print(">>>> Overwrite args...")

    Initializer.manual_seed(opt.seed)
    print(">> Constructing the dataset...")
    if opt.trainval == 0:
        trainset = VQADataset(opt.data_path, opt.data_name, "train", opt.img_path, opt.img_type, "trainval")
        if False:
            trainLoader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                     num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate,
                                     batch_sampler=BatchSampler)
        else:
            trainLoader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                     num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate)

        valset = VQADataset(opt.data_path, opt.data_name, "val", opt.img_path, opt.img_type, "trainval")
        if False:
            valLoader = DataLoader(valset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                   num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate,
                                   batch_sampler=BatchSampler)
        else:
            valLoader = DataLoader(valset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                   num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate)
    else:
        trainset = VQADataset(opt.data_path, opt.data_name, "trainval", opt.img_path, opt.img_type, "trainval")
        trainLoader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                 num_workers=opt.num_workers, pin_memory=True, collate_fn=default_collate,
                                 batch_sampler=BatchSampler)

        valset = None
        valLoader = None

    print(">> Building model...")
    word_embedded = LargeEmbedding(len(trainset.idx2word), 300, padding_idx=0, devices=opt.gpus)
    word_embedded.load_pretrained_vectors(opt.word_vectors)

    idx2ans = trainset.idx2ans
    num_ans = trainset.ans_pool.shape[0]

    if opt.arch == "DCNWithRCNN":
        model = DCNWithRCNN(opt, num_ans)
    elif opt.arch == "DCN":
        model = DCN(opt, num_ans)

    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    if len(opt.gpus) >= 1:
        model = model.cuda(opt.gpus[0])
        criterion = criterion.cuda(opt.gpus[0])

    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, opt.gpus, dim=0)
    model.word_embedded = word_embedded

    params = list(filter(lambda x: x.requires_grad, model.parameters()))

    optimizer = FixedAdam(opt, params)
    scheduler = StepScheduler(opt, optimizer)

    params = model.module.parameters() if len(opt.gpus) > 1 else model.parameters()
    print(">> Number of trained parameters:", sum(param.numel() for param in params if param.requires_grad))

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print(">>>> Loaded checkpoint: {} - epoch {}".format(opt.resume, checkpoint["last_epoch"]))

    timer = TimeMeter()
    timer.reset()
    try:
        with torch.cuda.device(opt.gpus[0]):
            trainModel(trainLoader, valLoader, model, criterion, optimizer, scheduler, checkpoint, idx2ans, opt)
    except KeyboardInterrupt:
        sys.exit("Training interrupted")

    print("It toke %.2f hours to train the network" % (timer.elapsed_time / 3600))

if __name__ == "__main__":
    args = get_train_config()
    params = vars(args)
    print("Parsed input parameters:")
    print(json.dumps(params, indent=2))
    main(args)
