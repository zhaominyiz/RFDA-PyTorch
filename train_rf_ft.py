import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import math
import time
import yaml
import argparse
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os.path as op
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from PIL import Image
import utils  # my tool box
import dataset
import gc
from net_rfda import RFDA
# Multi-State-Training
# torch.autograd.set_detect_anomaly(True)
def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_2G.yml', 
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )

    return opts_dict

def get_lr(lr,milestones,it,gamma):
    count=0
    for milestone in milestones:
        if(it>milestone):count+=1
    return lr*pow(gamma,count)

def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # ==========
    # init distributed training
    # ==========
    # opts_dict['train']['is_dist'] = False
    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank, 
            backend='nccl'
            )

    # TO-DO: load resume states if exists
    pass

    # ==========
    # create logger
    # ==========

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        if not os.path.exists(log_dir):
            utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'a')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========

    pass
    
    # ==========
    # fix random seed
    # ==========

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create train and val data prefetchers
    # ==========
    
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    radius = opts_dict['network']['radius']
    radius_real = 3
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    assert val_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    val_ds_cls = getattr(dataset, val_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'], 
        radius=radius
        )
    val_ds = val_ds_cls(
        opts_dict=opts_dict['dataset']['val'], 
        radius=radius
        )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds, 
        num_replicas=opts_dict['train']['num_gpu'], 
        rank=rank, 
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
        )
    val_sampler = None  # no need to sample val data

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds, 
        opts_dict=opts_dict, 
        sampler=train_sampler, 
        phase='train',
        seed=opts_dict['train']['random_seed']
        )
    val_loader = utils.create_dataloader(
        dataset=val_ds, 
        opts_dict=opts_dict, 
        sampler=val_sampler, 
        phase='val'
        )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
        opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
        opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)
    
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # ==========
    # create model
    # ==========

    model = RFDA(opts_dict=opts_dict['network'])

    model = model.to(rank)
    # if opts_dict['train']['is_dist']:
    #     model = DDP(model, device_ids=[rank])

    # ==========
    # define loss func & optimizer & scheduler & criterion
    # ==========

    # define loss func
    if opts_dict['train']['loss']['type'] == 'CharbonnierLoss':
        print("CharbonnierLoss")
        loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])
    # elif opts_dict['train']['loss']['type'] == 'L1':
    #     print("L1Loss")
    #     loss_func = nn.L1Loss()
    # elif opts_dict['train']['loss']['type'] == 'L2':
    #     print("MSELoss")
    #     loss_func = nn.MSELoss()
    # elif opts_dict['train']['loss']['type'] == 'L1FFT':
    #     print("L1FFTLoss")
    #     loss_func = utils.L1FFTLoss()
    # elif opts_dict['train']['loss']['type'] == 'L2FFT':
    #     print("L2FFTLoss")
    #     loss_func = utils.L2FFTLoss()
    # elif opts_dict['train']['loss']['type'] == 'L2SSIM':
    #     print("L2SSIMLoss")
    #     loss_func = utils.L2SSIMLoss()
    # elif opts_dict['train']['loss']['type'] == 'CharbFFT':
    #     print("CharbFFT")
    #     loss_func = utils.CharbFFTLoss()

    # TODO 
    # loss_func = torch.nn.MSELoss()
    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    # print(opts_dict['train']['optim'],"???")

    # optimizer = optim.Adam(model.parameters(),**opts_dict['train']['optim'])
    # fix the ffnet part
    base_params = filter(lambda p: id(p) not in list(map(id, model.qenet.parameters())),model.parameters())
    optimizer = optim.Adam(
        [
            {"params": base_params, "lr": opts_dict['train']['optim']["lr"]/20.0},
            {"params": model.qenet.parameters(), "lr": opts_dict['train']['optim']["lr"]}
        ],
        **opts_dict['train']['optim']
        )
    # define scheduler
    milestones=[]
    for milestone in opts_dict['train']['scheduler']['milestones']:
        milestones.append(int(milestone*num_iter))
    gamma=opts_dict['train']['scheduler']['gamma']
    opt_lr=opts_dict['train']['optim']['lr']
    set_milestones=set(milestones)

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()
    start_iter = 0
    # ==========
    # TO-DO: resume training & load pre-trained model
    # ==========

    
    # filename = '/data/myzhao/NTIRE21VQE/pretrain/R3L_l1fft69_new_lmdb_fix_rgb_s69_hq_no_reverse_ckp_600000.pt'
    # filename = 'ckp_495000.pt'
    # filename = op.join(
    #     "exp", opts_dict['train']['exp_name'], filename
    #     )
    # filename = '/remote-home/myzhao/MM_CKPS/R3_QP37.pt'
    filename = 'exp/rf_only/ckp_5000.pt'
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    
    # start_iter = checkpoint['num_iter_accum']
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # if opts_dict['train']['scheduler']['is_on']:
    #     for i in range(start_iter):
    #         scheduler.step()
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # if not 'qenet' in k:
            #     continue
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict,strict=False)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    now_lr=get_lr(opt_lr,milestones,start_iter,gamma)
    # print(opt_lr,milestones,start_iter,gamma)
    # print("now_lr",now_lr)
    # if (opts_dict["network"]['qenet']['netname'] != 'C2CNET'):
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = now_lr
    # optimizer.param_groups[0]["lr"] = now_lr
    # optimizer.param_groups[0]["lr"] = now_lr/10.0
    # optimizer.param_groups[1]["lr"] = now_lr
    # print("=> loaded checkpoint Success ")
    
    torch.autograd.set_detect_anomaly(True)
    # start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"val sequence: [{val_num}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # Test LR
    # for i in range(num_iter):
    #     optimizer.step()
    #     scheduler.step()
    #     lr = optimizer.param_groups[0]['lr']
    #     if i%1000==0:
    #         print(i,lr,i/num_iter)
    # os._exit(233)
    # ==========
    # evaluate original performance, e.g., PSNR before enhancement
    # ==========
    vid_num = val_ds.get_vid_num()
    if opts_dict['train']['pre-val'] and rank == 0:
        msg = f"\n{'<' * 10} Pre-evaluation {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        per_aver_dict = {}
        for i in range(vid_num):
            per_aver_dict[i] = utils.Counter()
        pbar = tqdm(
                total=val_num, 
                ncols=opts_dict['train']['pbar_len']
                )

        # fetch the first batch
        val_prefetcher.reset()
        val_data = val_prefetcher.next()

        while val_data is not None:
            # get data
            gt_data = val_data['gt']  # (B [RGB] H W)
            lq_data = val_data['lq']  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            b, _, _, _, _  = lq_data.shape
           
            batch_perf = np.mean(
                [criterion(lq_data[i,radius,...], gt_data[i]) for i in range(b)]
                )  # bs must be 1!
            
            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)

            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s}".format(name_vid, batch_perf, unit)
                )
            pbar.update()

            # fetch next batch
            val_data = val_prefetcher.next()

        pbar.close()

        # log
        ave_performance = np.mean([
            per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
            ])
        msg = "> ori performance: [{:.3f}] {:s}".format(ave_performance, unit)
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch

    # ==========
    # start training + validation (test)
    # ==========

    model.train()
    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        # train this epoch
        while train_data is not None:

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt']  # (B 3 [RGB] H W)
            lq_data = train_data['lq']#.to(rank)  # (B T [RGB] H W)
            # print(gt_data.size(),"vs",lq_data.size())
            b, t, c, _, _  = lq_data.shape

            input_data = lq_data
            loss_sum = 0
            enhanced_datas = model(input_data,rank=rank)
            optimizer.zero_grad()  # zero grad
            
            loss = 0
            # print(len(enhanced_datas),enhanced_datas[0].size(),'vs',gt_data.size())
            for i in range(t):
                loss = loss + torch.mean(torch.stack(
                    [loss_func(enhanced_datas[i][j,...], gt_data[j,i,...]) for j in range(b)]
                    ))
            loss.backward()  # cal grad
            optimizer.step()  # update parameters
            # os._exit(233)
            # update learning rate
            
            if (num_iter_accum % interval_print == 0) and (rank == 0):
                now_lr = get_lr(opt_lr, milestones, num_iter_accum, gamma)
                # if (now_lr != optimizer.param_groups[0]['lr']):
                #     for param_group in optimizer.param_groups:
                #         param_group["lr"] = now_lr
                    # optimizer.param_groups[0]["lr"] = now_lr/10.0
                    # optimizer.param_groups[1]["lr"] = now_lr
                    # optimizer.param_groups[2]["lr"] = now_lr/10.0
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}]".format(
                        lr*1e4, loss_item
                        )
                    )
                print(msg)
                log_fp.write(msg + '\n')
            
            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pt"
                    )
                state = {
                    'num_iter_accum': num_iter_accum, 
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    }
                torch.save(state, checkpoint_save_path)
                # print("continue")
                # continuecontinue
                # validation
                with torch.no_grad():
                    per_aver_dict = {}
                    for index_vid in range(vid_num):
                        per_aver_dict[index_vid] = utils.Counter()
                    pbar = tqdm(
                            total=7980, 
                            ncols=opts_dict['train']['pbar_len']
                            )
                
                    # train -> eval
                    model.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()
                    
                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt']  # (B T [RGB] H W)
                        lq_data = val_data['lq']  # (B T [RGB] H W)
                        # print(gt_data)
                        # os._exit(2333)
                        index_vid = val_data['index_vid'].item()
                        name_vid = val_data['name_vid'][0]  # bs must be 1!
                        b, t, c, _, _  = lq_data.shape
                        # input_data = torch.cat(
                        #     [lq_data[:,:,i,...] for i in range(c)], 
                        #     dim=1
                        #     )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
                        if t>=100:
                            enhanced_data = []
                            for i in range(0,t,100):
                                last = min(i+100,t)
                                enhanced_data+=model(lq_data[:,i:last,...],rank=rank,is_test=False)
                        else:
                            enhanced_data = model(lq_data,rank=rank,is_test=False)
                        # print(len(enhanced_data))
                        for i in range(t):
                            # print(enhanced_data[i].size(),'vs',gt_data[:,i,...].size())
                            batch_perf = np.mean(
                                [criterion(enhanced_data[i][j,...].to(rank), gt_data[j,i,...].to(rank)) for j in range(b)]
                                ) # bs must be 1!

                            # display
                            pbar.set_description(
                                "{:s}: [{:.3f}] {:s}"
                                .format(name_vid, batch_perf, unit)
                                )
                            pbar.update()

                            # log
                            per_aver_dict[index_vid].accum(volume=batch_perf)
                        del enhanced_data
                            # torch.cuda.empty_cache()
                        # fetch next batch
                        torch.cuda.empty_cache()
                        val_data = val_prefetcher.next()
                        gc.collect()
                        # return
                    # end of val
                    pbar.close()

                    # eval -> train
                    model.train()

                # log
                ave_per = np.mean([
                    per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
                    ])
                msg = (
                    "> model saved at {:s}\n"
                    "> ave val per: [{:.3f}] {:s}"
                    ).format(
                        checkpoint_save_path, ave_per, unit
                        )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

        # end of this epoch (training dataloader exhausted)

    # end of all epochs

    # ==========
    # final log & close logger
    # ==========

    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
        print(msg)
        log_fp.write(msg + '\n')
        
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        
        log_fp.close()


if __name__ == '__main__':
    main()
    
