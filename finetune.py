import sys
import os
sys.path.append('.')       # 添加当前目录
sys.path.append('..')      # 添加上级目录
os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR ,CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb, AdamW
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint, load_components_from_pretrained
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from utils.make_master_file import DATASET_DICT
from models.fno import FNO2d
from MoEPOT.models.moepot import MoEPOTNet
from models.MoE_conv import MoEImage
import pickle


torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

### currently no influence
parser.add_argument('--model', type=str, default='MoEPOT')
parser.add_argument('--dataset',type=str, default='ns2d')

parser.add_argument('--train_paths',nargs='+', type=str, default=[
  'ns2d_fno_1e-5',
#   'ns2d_fno_1e-3',
#    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
#   'swe_pdb',
#   'dr_pdb',
#   'cfdbench'
])
parser.add_argument('--test_paths',nargs='+',type=str, default=[
 'ns2d_fno_1e-5',
#   'ns2d_fno_1e-3',
#    'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
#   'swe_pdb',
#   'dr_pdb',
#   'cfdbench'
])
parser.add_argument('--resume_path',type=str, default='./logs_pretrain/checkpoints/model_moepot_tiny.pth')
parser.add_argument('--ntrain_list', nargs='+', type=int, default=[1000])
parser.add_argument('--cls_id',type = int ,default = 0)   # 确定要计算误差的数据集,一个数据集时没有影响
parser.add_argument('--data_weights',nargs='+',type=int, default=[1,1,1,1,1,1])
parser.add_argument('--use_writer', action='store_true',default= False) # 记录结果

parser.add_argument('--res', type=int, default=128)
parser.add_argument('--noise_scale',type=float, default=0)  # finetune时，噪声为0

### shared params 
parser.add_argument('--width', type=int, default=512)    # width
parser.add_argument('--n_layers',type=int, default=4)  # n_layers = 4
parser.add_argument('--act',type=str, default='gelu')

### GNOT params
parser.add_argument('--max_nodes',type=int, default=-1)

### FNO params
parser.add_argument('--modes', type=int, default=32)    # modes
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=0)


### Params
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=4)         
parser.add_argument('--mlp_ratio',type=int, default=1)        
parser.add_argument('--out_layer_dim', type=int, default=32)

parser.add_argument('--batch_size', type=int, default=20)    
parser.add_argument('--epochs', type=int, default=500)       
parser.add_argument('--lr', type=float, default=0.001)      
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb','adamw'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.9)
parser.add_argument('--lr_method',type=str, default='cycle') #  cycle
parser.add_argument('--grad_clip',type=float, default=1000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=100)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="1")
parser.add_argument('--comment',type=str, default="Tft_ns1e-5_ours")    # Save file name
parser.add_argument('--log_path',type=str,default='')


### finetuning parameters
parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=6)    # The default dataset consists of 6 categories, consistent with the amount used during pre-training
parser.add_argument('--load_components',nargs='+', type=str, default=['blocks','pos','time_agg'])
parser.add_argument('--moe_loss_weight',type=float,default=0.1)
parser.add_argument('--is_finetune',action='store_true',default=True)  # in finetune stage

args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))

print(f"Current working directory: {os.getcwd()}")


#冻结函数
def freeze_all_moe_gating_and_feature(model):
    """
    finetune阶段冻结MoE的门控和特征提取器
    """
    for module in model.modules():
        if isinstance(module, MoEImage):
            # 1. gating
            module.gating.eval()
            for p in module.gating.parameters():
                p.requires_grad = False

            # 2. feature extractor
            module.feature_extractor.eval()
            for p in module.feature_extractor.parameters():
                p.requires_grad = False



################################################################
# load data and dataloader
################################################################
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)


train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in=args.T_in, t_ar=args.T_ar, normalize=False, train=True, data_weights=args.data_weights, n_channels=args.n_channels)
test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels, t_in=args.T_in, t_ar=-1, normalize=False, train=False) for _, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8) for test_dataset in test_datasets]
ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))
################################################################
# load model
################################################################
if args.model == 'MoEPOT':  # this one
    model = MoEPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=args.n_class,is_finetune=args.is_finetune).to(device)
else:
    raise NotImplementedError

if args.resume_path:
    print('Loading models and fine tune from {}'.format(args.resume_path))
    load_model_from_checkpoint(model, torch.load(args.resume_path,map_location='cuda:{}'.format(args.gpu))['model'])

#### set optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
if args.opt == 'lamb':
    optimizer = Lamb(trainable_params, lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False, weight_decay=1e-4)
elif args.opt == 'adamw':
    optimizer = AdamW(trainable_params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-4)
else:
    optimizer = Adam(trainable_params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * len(train_loader),mode='triangular2', cycle_momentum=False)
elif args.lr_method == 'Cos':
    scheduler = CosineAnnealingLR(
    optimizer,
    T_max=args.epochs*len(train_loader), 
    eta_min=1e-6    # Minimum learning rate
)
else:
    raise NotImplementedError

comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
model_path = log_path + '/model.pth'  # Save the model path in the logs folder
if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + '/logs.txt', 'w+',buffering=1)
    json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp

else:
    writer = None
print("is_finetune:",args.is_finetune)
print(model)
count_parameters(model)

################################################################
# Main function for pretraining
################################################################
myloss = SimpleLpLoss(size_average=False)
iter = 0
for ep in range(args.epochs):
    model.train()
    freeze_all_moe_gating_and_feature(model)
    
    t1 = t_1 = default_timer()
    t_load, t_train = 0., 0.
    train_l2_step = 0
    train_l2_full = 0
    cls_total, cls_correct, cls_acc = 0, 0, 0.
    loss_previous = np.inf

    for xx, yy, msk, cls in train_loader:
        t_load += default_timer() - t_1
        t_1 = default_timer()

        loss, cls_loss ,loss_gate_total = 0. , 0. ,0.
        xx = xx.to(device)  ## B, n, n, T_in, C
        yy = yy.to(device)  ## B, n, n, T_ar, C
        msk = msk.to(device)
        cls = cls.to(device)
        mask_cls = (cls.squeeze() == args.cls_id) # [B]
        if mask_cls.sum() == 0:
            continue


        ## auto-regressive training loop, support 1. noise injection, 2. long rollout backward, 3. temporal bundling prediction
        for t in range(0, yy.shape[-2], args.T_bundle):
            y = yy[..., t:t + args.T_bundle, :]

            ### auto-regressive training
            xx = xx + args.noise_scale *torch.sum(xx**2, dim=(1,2,3),keepdim=True)**0.5 * torch.randn_like(xx)
            im, cls_pred , _ = model(xx) # Finetune stage no longer requires moe loss
            loss += myloss(im[mask_cls], y[mask_cls], mask=msk[mask_cls])
            ### classification

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-2)
                

            xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

        train_l2_step += loss.item()
        l2_full = myloss(pred[mask_cls], yy[mask_cls], mask=msk[mask_cls])
        train_l2_full += l2_full.item()

        optimizer.zero_grad() # NOTE: this line is commented NEED TO BE CHECKED
        total_loss = loss
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        
        

        train_l2_step_avg, train_l2_full_avg = train_l2_step / args.ntrain_list[args.cls_id] / (yy.shape[-2] / args.T_bundle), train_l2_full / args.ntrain_list[args.cls_id]
        # cls_acc = cls_correct / cls_total
        iter += 1
        if args.use_writer:
            writer.add_scalar("train_loss_step", loss.item()/(xx.shape[0] * yy.shape[-2] / args.T_bundle), iter)
            writer.add_scalar("train_loss_full", l2_full / xx.shape[0], iter)

            ## reset model
            if loss.item() > 10 * loss_previous : # or (ep > 50 and l2_full / xx.shape[0] > 0.9):
                print('loss explodes, loading model from previous epoch')
                checkpoint = torch.load(model_path,map_location='cuda:{}'.format(args.gpu))
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint["optimizer"])
                loss_previous = loss.item()

        t_train += default_timer() -  t_1
        t_1 = default_timer()

    test_l2_fulls, test_l2_steps = [], []
    model.eval()
    with torch.no_grad():
        for id, test_loader in enumerate(test_loaders):
            test_l2_full, test_l2_step = 0, 0
            for xx, yy, msk, _ in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                msk = msk.to(device)

                for t in range(0, yy.shape[-2], args.T_bundle):
                    y = yy[..., t:t + args.T_bundle, :]
                    im, _ , _ = model(xx)
                    loss += myloss(im, y, mask=msk)

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -2)

                    xx = torch.cat((xx[..., args.T_bundle:,:], im), dim=-2)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred, yy, mask=msk)

            test_l2_step_avg, test_l2_full_avg = test_l2_step / ntests[id] / (yy.shape[-2] / args.T_bundle), test_l2_full / ntests[id]
            test_l2_steps.append(test_l2_step_avg)
            test_l2_fulls.append(test_l2_full_avg)
            if args.use_writer:
                writer.add_scalar("test_loss_step_{}".format(test_paths[id]), test_l2_step_avg, ep)
                writer.add_scalar("test_loss_full_{}".format(test_paths[id]), test_l2_full_avg, ep)

    if args.use_writer:
        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)

    t_test = default_timer() - t_1
    t2 = t_1 = default_timer()
    lr = optimizer.param_groups[0]['lr']
    print('epoch {}, time {:.5f}, lr {:.2e}, train l2 step {:.5f} train l2 full {:.5f}, test l2 step {} test l2 full {}, time train avg {:.5f} load avg {:.5f} test {:.5f}'.format(
        ep, 
        t2 - t1, 
        lr,
        train_l2_step_avg, 
        train_l2_full_avg,
        ', '.join(['{:.5f}'.format(val) for val in test_l2_steps]),
        ', '.join(['{:.5f}'.format(val) for val in test_l2_fulls]), 
        # cls_acc, 
        t_train / len(train_loader), 
        t_load / len(train_loader), 
        t_test))




