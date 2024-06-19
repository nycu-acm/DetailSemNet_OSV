import sys
import json
from PIL import Image
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import timm
from model_v3 import ViT_for_OSV_DSNet
from sig_dataloader import SigDataset_BH as SigDataset_BH_v1
from sig_dataloader import SigDataset, SigDataset_GPDS
from sig_dataloader_v2 import SigDataset_BH as SigDataset_BH_v2
from sig_dataloader_v2 import SigDataset_CEDAR
from module.scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import auc
####################
# with model_v3.py #
####################
# REPRODUCIBILITY
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size') # train: 4, test: 16
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--epoch', type=int, default=400) #400
parser.add_argument('--name', default="demo", help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')
parser.add_argument('--data', type=str, default="./../ChiSig", help='dataset path')
parser.add_argument('--data_mode', type=str, default="normalized", help='data path') # [normalized, cropped, centered, left]
parser.add_argument('--test_only', action='store_true', help='test mode')
parser.add_argument('--shift', action='store_true', help='shift input image during train/test')
parser.add_argument('--model_type', type=str, default="DSNet", help='model type') # v1: only vit, v2: vit + HelixTransformer, v3: vit + CPD
parser.add_argument('--patch_type', type=str, default="v0", help='patch type') # v0: Conv. split, t2t: tokens2token module
parser.add_argument('--convert_type', type=str, default="L", help='image convert type (pil)')
parser.add_argument('--fs', action='store_true', help='use few-shot setting')
parser.add_argument('--part', action='store_true', help='use self divided partition')
###Loss###
parser.add_argument('--loss', type=str, default="con", help='select different loss') #['bce', 'con']

parser.add_argument('--comment', type=str, default="", help='some comment')

parser.add_argument('--emd', action='store_true', help='run emd')
parser.add_argument('--temp', type=float, default=1., help='set emd temp hyperparameter')
parser.add_argument('--dis_type', type=str, default='cos', help='set local emd distance type')
parser.add_argument('--gol_dis', type=str, default='l2', help='set global distance type') # 'l2', 'cos'
parser.add_argument('--mar_type', type=str, default='uniform', help='set emd weighting') # ['uniform', 'different']

opt = parser.parse_args()

torch.backends.cudnn.benchmark = True

def compute_accuracy_roc(predictions, labels, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    if step is None:
        step = 0.00005

    max_acc, min_frr, min_far = 0.0, 1.0, 1.0
    min_dif = 1.0
    d_optimal = 0.0
    tpr_arr, fpr_arr, far_arr, frr_arr, d_arr = [], [], [], [], []
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d     # pred = 1
        idx2 = predictions.ravel() > d      # pred = 0

        tp = float(np.sum(labels[idx1] == 1))
        tn = float(np.sum(labels[idx2] == 0))
        
        tpr = tp / nsame
        tnr = tn / ndiff

        frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff

        tpr_arr.append(tpr)
        far_arr.append(far)
        frr_arr.append(frr)
        d_arr.append(d)
        
        acc = (tp+tn) / (nsame+ndiff)

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            
            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff
        
        if abs(far-frr) < min_dif:
            min_dif = abs(far-frr)
            d_optimal_diff = d
            
            # FRR, FAR metrics
            min_dif_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_dif_far = float(np.sum(labels[idx1] == 0)) / ndiff
            
    print("EER: {} @{}".format((min_dif_frr+min_dif_far)/2.0, d_optimal_diff))
    metrics = {"best_acc" : max_acc, "best_frr" : min_frr, "best_far" : min_far, "tpr_arr" : tpr_arr, "far_arr" : far_arr, "frr_arr" : frr_arr, "d_arr": d_arr}
    return metrics, d_optimal

def get_pct_accuracy(pred: Variable, target, path=None) -> int:
    if opt.loss == 'con':
        hard_pred = (pred < 0.5).int()
    elif opt.loss == 'bce':
        hard_pred = (pred > 0.5).int()
    else:
        return NotImplementedError
    
    correct = (hard_pred == target).sum().data
    accuracy = float(correct) / target.size()[0]
    
    accuracy = int(accuracy * 100)
    return accuracy

def train(opt):
    if 'BHSig260' in opt.data:
        sigdataset_train = SigDataset_BH_v1(opt, opt.data, train=True, image_size=opt.imageSize, mode=opt.data_mode)
        sigdataset_test = SigDataset_BH_v1(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    elif 'ChiSig' in opt.data:
        sigdataset_train = SigDataset(opt.data, train=True, image_size=opt.imageSize, convert_type=opt.convert_type)
        sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize, convert_type=opt.convert_type)
    elif 'CEDAR' in opt.data:
        sigdataset_train = SigDataset_CEDAR(opt, opt.data, train=True, image_size=opt.imageSize, mode=opt.data_mode)
        sigdataset_test = SigDataset_CEDAR(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)    
    elif 'GPDS' in opt.data:
        sigdataset_train = SigDataset_GPDS(opt, opt.data, train=True, image_size=opt.imageSize, mode=opt.data_mode)
        sigdataset_test = SigDataset_GPDS(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    else:
        print('not implement')
        return NotImplementedError

    train_loader = DataLoader(sigdataset_train, batch_size=opt.batchSize, shuffle=True, pin_memory=True, num_workers=16)
    test_loader = DataLoader(sigdataset_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=16)

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)
    os.makedirs(models_path, exist_ok=True)
    with open(os.path.join(models_path, 'args.txt'),'w') as f:
        f.write(' '.join(str(x) for x in sys.argv))
        json.dump(opt.__dict__,f,indent=4)
    
    model = ViT_for_OSV_DSNet(opt=opt)
    model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=1e-4)
    max_epoch = opt.epoch
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
    scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=8, total_epoch=10,after_scheduler=scheduler_cosine)
    
    best_validation_loss = None
    saving_threshold = 0.9
    last_saved = datetime.utcnow()
    save_every = timedelta(hours=2)

    writer = SummaryWriter(log_dir='runs/{}'.format(opt.name))

    iteration = 0
    test_len = len(train_loader)
    if opt.fs:
        test_len = 50
    training_loss = 0.0
    train_acc = val_acc = 0.0
    global_step = 0
    for epoch in range(opt.epoch): # 400
        for index, (X, Y) in enumerate(tqdm(train_loader)):
            model.train()
            X = X.view(-1,2,opt.imageSize,opt.imageSize)
            Y = Y.view(-1,1)
            X, Y = X.cuda(), Y.cuda()
            pred, loss = model(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.data
            if isinstance(pred, tuple):
                train_acc += get_pct_accuracy(pred[0], Y)
            else:
                train_acc += get_pct_accuracy(pred, Y)

            if (iteration) % test_len == 0:
                iteration = 0
                validation_loss = 0.0
                for X_val, Y_val in tqdm(test_loader):
                    model.eval()
                    with torch.no_grad():
                        # validate your model
                        X_val = X_val.view(-1,2,opt.imageSize,opt.imageSize).cuda()
                        Y_val = Y_val.view(-1,1).cuda()

                        pred_val, loss_val = model(X_val, Y_val)
                            
                        validation_loss += loss_val.data
                        
                        if isinstance(pred_val, tuple):
                            val_acc += get_pct_accuracy(pred_val[0], Y_val)
                        else:
                            val_acc += get_pct_accuracy(pred_val, Y_val)
                
                training_loss /= (float)(test_len)
                validation_loss /= (float)(len(test_loader))
                train_acc /= (float)(test_len)
                val_acc /= (float)(len(test_loader))
                writer.add_scalar("training_loss", training_loss, epoch)
                writer.add_scalar("train_acc", train_acc, epoch)
                writer.add_scalar("validation_loss", validation_loss, epoch)
                writer.add_scalar("val_acc", val_acc, epoch)


                print("Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}".format(
                    epoch, train_acc, training_loss, val_acc, validation_loss
                ))

                if best_validation_loss is None and epoch > 1:
                    best_validation_loss = validation_loss

                if best_validation_loss is not None and best_validation_loss > (saving_threshold * validation_loss):
                    print("Significantly improved validation loss from {} --> {}. Saving...".format(
                        best_validation_loss, validation_loss
                    ))
                    model.save_to_file(os.path.join(models_path, "{}_{:.4f}.pt".format(epoch, validation_loss.item())))
                    best_validation_loss = validation_loss
                    last_saved = datetime.utcnow()
                
                training_loss = 0.0
                train_acc = val_acc = 0.0
            iteration += 1
        scheduler.step(epoch)
    writer.close()

def test(opt):
    if 'BHSig260' in opt.data:
        sigdataset_test = SigDataset_BH_v2(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    elif 'ChiSig' in opt.data:
        sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize, convert_type=opt.convert_type)
    elif 'CEDAR' in opt.data:
        sigdataset_test = SigDataset_CEDAR(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    elif 'GPDS' in opt.data:
        sigdataset_test = SigDataset_GPDS(opt, opt.data, train=False, image_size=opt.imageSize, mode=opt.data_mode)
    else:
        return NotImplementedError
    
    print(len(sigdataset_test))
    test_loader = DataLoader(sigdataset_test, batch_size=opt.batchSize, shuffle=False) #pin_memory=True, num_workers=16
    
    models_path = os.path.join("saved_models", opt.name)
    model = ViT_for_OSV_DSNet(opt=opt)
    model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
    
    validation_loss = val_acc = 0.0
    np_loss = np.zeros(shape=(0,))
    gt_loss = np.zeros(shape=(0,))

    for item in tqdm(test_loader):
        if len(item) == 2:
            X_val, Y_val = item
        elif len(item) == 3:
            X_val, Y_val, path = item
        else:
            return
        
        model.eval()
        with torch.no_grad():
            # validate your model
            X_val = X_val.view(-1,2,opt.imageSize,opt.imageSize).cuda()
            Y_val = Y_val.view(-1,1).cuda()
            
            pred_val, loss_val = model(X_val, Y_val)
            validation_loss += loss_val.data
            if isinstance(pred_val, tuple):
                val_acc += get_pct_accuracy(pred_val[0], Y_val)
            else:
                val_acc += get_pct_accuracy(pred_val, Y_val)

            if isinstance(pred_val, tuple):
                pred_val_ = pred_val[0] + opt.temp*pred_val[1]
                np_loss = np.append(np_loss, pred_val_.cpu().detach().numpy())
            else:
                np_loss = np.append(np_loss, pred_val.cpu().detach().numpy())
            gt_loss = np.append(gt_loss, Y_val.cpu().detach().numpy())
    
    validation_loss /= (float)(len(test_loader))
    val_acc /= (float)(len(test_loader))

    print("Validation: Acc={}%, Loss={}".format(val_acc, validation_loss))
    
    metrics, thresh_optimal = compute_accuracy_roc(np_loss, gt_loss, step=5e-5)

    #print("d optimal: {}".format(thresh_optimal))
    print("Metrics obtained: \n" + '-'*50)
    print(f"Acc: {metrics['best_acc'] * 100 :.4f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.4f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.4f} %")
    print('-'*50)
    print("AUC: {}".format(auc(np.array(metrics['far_arr']), np.array(metrics['tpr_arr']))))
    return

def main() -> None:
    if not opt.test_only:
        train(opt)
    else:
        test(opt)

if __name__ == "__main__":
    main()
