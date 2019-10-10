#coding:utf-8

import os
import json

import torch
from dense_coattn.utils import (AverageMeter, Initializer, StopwatchMeter,
                                TimeMeter, extract_statedict, move_to_cuda,
                                save_checkpoint)


def evalEpoch(epoch, dataloader, model, criterion, opt, writer):
    torch.set_grad_enabled(False)
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()

    #for batch in dataloader:
    for i, batch in enumerate(dataloader):
        img_info, ques, ques_mask, ans_idx = move_to_cuda(batch[:-1], devices=opt.gpus)
        ques = model.word_embedded(ques).detach()

        if False:
            img, img_mask = img_info
        else:
            img = img_info
            img_mask = img_info


        score = model(img, ques, img_mask, ques_mask)
        loss = criterion(score, ans_idx)
        accuracy = evaluate(score, ans_idx)

        losses.update((loss.item() / opt.batch_size))
        accuracies.update(accuracy.item())

        if (epoch * len(dataloader) + i + 1) >= opt.num_iter and opt.num_iter > 0:
            break

    if writer is not None:
        writer.add_scalar("epoch/val_loss", losses.avg, global_step=epoch)
        writer.add_scalar("epoch/val_accuracy", accuracies.avg, global_step=epoch)

    return losses.avg, accuracies.avg


def vqaEval(dataloader, model, criterion, idx2ans, opt):
    criterion.reduction = 'none'
    torch.set_grad_enabled(False)
    model.eval()
    result = []
    dataset_length = len(dataloader.dataset)

    if os.path.exists(os.path.join(opt.result_file, '{}.json'.format(opt.model))):
        result = json.load(open(os.path.join(opt.result_file, '{}.json'.format(opt.model))))
    else:
        for i, batch in enumerate(dataloader):
            ques_idx = batch[-1]
            img_info, ques, ques_mask, ans_idx = move_to_cuda(batch[:-1], devices=opt.gpus)
            ques = model.word_embedded(ques)
            if False:
                img, img_mask = img_info
            else:
                img = img_info
                img_mask = img_info

            score = model(img, ques, img_mask, ques_mask)
            loss = criterion(score, ans_idx)
            _, inds = torch.sort(score, dim=1, descending=True)

            for j in range(min(ques_idx.size(0), dataset_length - i * opt.batch_size)):
                result.append({"question_id": ques_idx[j].item(),
                               "answer": idx2ans[inds[j, 0].item()],
                               "entropy": loss[j].mean().item()})

        json.dump(result, open(os.path.join(opt.result_file, '{}.json'.format(opt.model)), "w"))
    vqa = VQA(opt.ann_file, opt.ques_file)
    vqa.load_result(result)
    vqa_eval = VQAEval(vqa)
    vqa_eval.evaluate()
    vqa_eval.compute_entropy()

    print("\n")
    print(">>>> Overall Accuracy is: %.02f\n" % (vqa_eval.accuracy["overall"]))
    print(">>>> Per Question Type Accuracy & Entropy is the following:")
    for quesType in vqa_eval.accuracy["per_questype"]:
        print("%s : %.02f, %.04f" % (quesType,
                                     vqa_eval.accuracy["per_questype"][quesType],
                                     vqa_eval.entropy["per_questype"][quesType]))
    print("\n")
    print(">>>> Per Answer Type Accuracy & Entropy is the following:")
    for ansType in vqa_eval.accuracy["per_anstype"]:
        print("%s : %.02f, %.04f" % (ansType,
                                     vqa_eval.accuracy["per_anstype"][ansType],
                                     vqa_eval.entropy["per_anstype"][ansType]))
    print("\n")
    json.dump({"accuracy": vqa_eval.accuracy, "entropy": vqa_eval.entropy},
              open(os.path.join(opt.result_file, '{}_acc.json'.format(opt.model)), "w"))
