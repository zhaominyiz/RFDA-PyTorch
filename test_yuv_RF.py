# @ author: Minyi Zhao
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import utils
import dataset
from net_rfda import RFDA
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

needSSIM = True
def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']
    PSNRS = []
    SSIMS = []
    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict['test'])}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()
    radius_real = 3
    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'], 
        radius=radius
        )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds, 
        opts_dict=opts_dict, 
        sampler=test_sampler, 
        phase='val'
        )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = RTVQE(opts_dict=opts_dict['network'])

    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')
    checkpoint_save_path = '/remote-home/myzhao/MM_CKPS/Final_QP37.pt'
    checkpoint = torch.load(checkpoint_save_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    
    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # ==========
    # define criterion
    # ==========

    # define criterion
    assert opts_dict['test']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()
    # ==========
    # validation
    # ==========
                
    # create timer
    total_timer = utils.Timer()

    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    name_vid_dict = dict()
    if needSSIM:
        per_aver_dict_ssim = dict()
        ori_aver_dict_ssim = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils.Counter()
        ori_aver_dict[index_vid] = utils.Counter()
        if needSSIM:
            per_aver_dict_ssim[index_vid] = utils.Counter()
            ori_aver_dict_ssim[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=7980, 
        ncols=opts_dict['test']['pbar_len']
        )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():
        while val_data is not None:
            # get data
            gt_data = val_data['gt']  # (B [RGB] H W)
            lq_data = val_data['lq']  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            
            b, t, c, _, _  = lq_data.shape

            for i in range(t):
                neighbor_list = list(range(i - radius_real, i + radius_real + 1))
                neighbor_list = list(np.clip(neighbor_list, 0, t - 1))
                frm_list = []
                # print(neighbor_list,'vs',lq_data.size())
                for _i in neighbor_list:
                    # print("i=",i)
                    # print(lq_data[:,int(i),:...].size())
                    frm_list.append(lq_data[:,int(_i),...])
                frm_list = torch.cat(frm_list,1).cuda()
                if i==0:
                    # print("first size",first.size())
                    enhanced_data,hint = model(frm_list)  # (B [RGB] H W)
                else:
                    enhanced_data,hint = model(frm_list,hint)

                # eval
                batch_ori = criterion(lq_data[0, i, ...].cuda(), gt_data[0,i,...].cuda())
                batch_perf = criterion(enhanced_data[0], gt_data[0,i,...].cuda())
                enhanced_data = enhanced_data[0].cpu().squeeze().numpy()
                lq_data2 = lq_data[0, i, ...].cpu().squeeze().numpy()
                gt_data2 = gt_data[0,i].cpu().squeeze().numpy()
                # print(lq_data.shape,"vs",gt_data.shape)
                # batch_ori = peak_signal_noise_ratio(lq_data, gt_data, data_range=1.0)
                # batch_perf = peak_signal_noise_ratio(enhanced_data, gt_data, data_range=1.0)
                PSNRS.append(batch_perf-batch_ori)
                if needSSIM:
                    batch_ori_ssim = structural_similarity(lq_data2, gt_data2, data_range=1.0)
                    batch_perf_ssim = structural_similarity(enhanced_data, gt_data2, data_range=1.0)
                    SSIMS.append(batch_perf_ssim-batch_ori_ssim)
                print("Ave PSNR:",sum(PSNRS)/len(PSNRS))
                if needSSIM:
                    print("Ave SSIM:",sum(SSIMS)/len(SSIMS)*100.0)
                # Calculate SSIM
                # display
                pbar.set_description(
                    "{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                    .format(name_vid, batch_ori, unit, batch_perf, unit)
                    )
                pbar.update()

                # log
                per_aver_dict[index_vid].accum(volume=batch_perf)
                ori_aver_dict[index_vid].accum(volume=batch_ori)
                if needSSIM:
                    per_aver_dict_ssim[index_vid].accum(volume=batch_perf_ssim)
                    ori_aver_dict_ssim[index_vid].accum(volume=batch_ori_ssim)
                if name_vid_dict[index_vid] == "":
                    name_vid_dict[index_vid] = name_vid
                else:
                    assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # fetch next batch
            val_data = test_prefetcher.next()
        
    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    log_fp.write(msg + '\n')
    for index_vid in range(test_vid_num):
        per = per_aver_dict[index_vid].get_ave()
        ori = ori_aver_dict[index_vid].get_ave()
        name_vid = name_vid_dict[index_vid]
        msg = "PSNR: {:s}: [{:.3f}] {:s} -> [{:.3f}] {:s} Delta:[{:.3f}]".format(
            name_vid, ori, unit, per, unit,(per-ori)
            )
        print(msg)
        log_fp.write(msg + '\n')
        if needSSIM:
            per_ssim = per_aver_dict_ssim[index_vid].get_ave()
            ori_ssim = ori_aver_dict_ssim[index_vid].get_ave()
            msg = "SSIM: {:s}: [{:.3f}] {:s} -> [{:.3f}] {:s} Delta:[{:.3f}] ".format(
                name_vid, ori_ssim*100.0, unit, per_ssim*100.0, unit,(per_ssim-ori_ssim)*100.0
                )
            print(msg)
            log_fp.write(msg + '\n')
    ave_per = np.mean([
        per_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori = np.mean([
        ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    if needSSIM:
        ave_per_ssim = np.mean([
            per_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
            ])
        ave_ori_ssim = np.mean([
            ori_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
            ])
    msg = (
        f"{'> ori: [{:.3f}] {:s}'.format(ave_ori, unit)}\n"
        f"{'> ave: [{:.3f}] {:s}'.format(ave_per, unit)}\n"
        f"{'> delta: [{:.3f}] {:s}'.format(ave_per - ave_ori, unit)}"
        )
    print(msg)
    log_fp.write(msg + '\n'+'SSIM:\n')
    log_fp.flush()
    if needSSIM:
        msg = (
            f"{'> ori: [{:.3f}] {:s}'.format(ave_ori_ssim*100.0, unit)}\n"
            f"{'> ave: [{:.3f}] {:s}'.format(ave_per_ssim*100.0, unit)}\n"
            f"{'> delta: [{:.3f}] {:s}'.format(ave_per_ssim*100.0 - ave_ori_ssim*100.0, unit)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # final log & close logger
    # ==========

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
    PSNRS = np.array(PSNRS)
    # print("STD",std(PSNRS),"PVD",pvd(PSNRS))
    # log_fp.write("STD: "+str(std(PSNRS))+"PVD:"+str(pvd(PSNRS)))
    log_fp.close()
    

if __name__ == '__main__':
    main()