from datasets.spectral import TestData
from torch.utils.data import DataLoader 
import torch 
import os 
import os.path as osp 
import numpy as np 
from utils import save_image,compare_ssim,compare_psnr
from opts import parse_args
from models.gap_network import GAP_CCoT
from utils import generate_masks,Logger
from image_utils import shift

def test(args,network,mask,mask_s,src_mask,logger,writer=None,epoch=1):

    mask, mask_s = generate_masks(args.mask_path)
    src_mask = mask.to(args.device)
    # mask_s = mask_s.to(args.device)
    mask = shift(src_mask,2).to(args.device)
    mask_s = torch.sum(mask,dim=0)
    network = network.eval()
    test_data = TestData(args.test_data_path,src_mask.cpu()) 
    test_data_loader = DataLoader(test_data,shuffle=False,batch_size=1)    

    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    out_list,gt_list = [],[]
    for iter,data in enumerate(test_data_loader):
        meas,gt = data 
        meas = meas.float().to(args.device)
        mask = mask.to(args.device)
        mask_s = mask_s.to(args.device)

        batch_size,frames,height,width = gt.shape

        gt = gt.float().numpy()
        # Phi_s = mask_s.expand([batch_size,height,width])
        Phi = mask.repeat([batch_size, 1, 1, 1])
        Phi_s = mask_s.repeat([batch_size, 1, 1])
       
        with torch.no_grad():
            out_pic_list = network(meas, Phi, Phi_s)
            # out_pic_list = network(meas)
            out_pic = out_pic_list[-1].cpu().numpy()
            psnr_t = 0
            ssim_t = 0
            for ii in range(batch_size):
                for jj in range(frames):
                    out_pic_p = out_pic[ii,jj, :, :]
                    gt_t = gt[ii,jj, :, :]
                    psnr_t += compare_psnr(gt_t,out_pic_p)
                    ssim_t += compare_ssim(gt_t,out_pic_p)
            psnr = psnr_t / (batch_size* frames)
            ssim = ssim_t / (batch_size* frames)
            psnr_list.append(np.round(psnr,4))
            ssim_list.append(np.round(ssim,4))
            out_list.append(out_pic)
            gt_list.append(gt)

    test_dir = osp.join(args.save_dir,"test")
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    for i,name in enumerate(test_data.scene_list):
        _name,_ = name.split(".")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        out = out_list[i]
        gt = gt_list[i]
        for j in range(out.shape[0]):
            image_name = osp.join(test_dir,"epoch_"+str(epoch)+"_"+_name+"_"+str(j)+".png")
            save_image(out[j],gt[j],image_name)
    if writer is not None:
        writer.add_scalar("psnr_mean",np.mean(psnr_list),epoch)
        writer.add_scalar("ssim_mean",np.mean(ssim_list),epoch)
    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict,ssim_dict 

if __name__=="__main__":
    args = parse_args()
    args.save_dir = "test_results"
    network = GAP_CCoT()
    network = network.to(args.device)
    log_dir = osp.join("test_results","log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    if args.checkpoints is not None:
        network.load_state_dict(torch.load(osp.join(args.checkpoints,"last.pth")),strict=False)
    mask, mask_s = generate_masks(args.mask_path)
    psnr_dict, ssim_dict = test(args,network,mask,mask_s,mask,logger)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
