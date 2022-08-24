from datasets.spectral import TrainData
from torch.utils.data import DataLoader
from models.gap_network import GAP_CCoT 
from utils import generate_masks, Logger, save_image
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from opts import parse_args
from test import test
import time
from image_utils import shift

def train(args,network,logger,mask,mask_s,writer=None):
    dataset = TrainData(args.train_data_path,mask.cpu())
    dist_sampler = None
    if args.distributed:
        dist_sampler = DistributedSampler(dataset,shuffle=True)
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            sampler=dist_sampler,num_workers=args.num_workers)
    else:
        train_data_loader = DataLoader(dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    criterion  = nn.MSELoss()
    # criterion  = nn.L1Loss()
    criterion = criterion.to(args.device)
    src_mask = mask.to(args.device)
    # mask_s = mask_s.to(args.device)
    mask = shift(src_mask,2).to(args.device)
    mask_s = torch.sum(mask,dim=0)
    
    rank = 0
    if args.distributed:
        rank = dist.get_rank()
    for epoch in range(args.epochs):
        
        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            
            meas,gt = data
            max =torch.max(meas)
            gt = gt.to(args.device)
            gt = gt.float()
            meas = meas.to(args.device)
            meas = meas.float()
            batch_size,frames,height,width = gt.shape
            Phi = mask.repeat(batch_size,1,1,1)
            Phi_s = mask_s.repeat(batch_size,1,1)
            optimizer.zero_grad()
            model_out = network(meas,Phi,Phi_s)
           
            loss = torch.sqrt(criterion(model_out[-1], gt))
            
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            if rank==0 and (iteration % args.show_step) == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info("epoch: {:<3d}, iter: {:<4d} lr: {:.6f}, loss: {:.5f}.".format(epoch,iteration,lr,loss.item()))
                writer.add_scalar("loss",loss.item(),epoch*len(train_data_loader) + iteration)
            if rank==0 and (iteration % args.save_image_step) == 0:
                if not osp.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                sing_out = model_out[-1][0].detach().cpu().numpy()
                sing_gt = gt[0].cpu().numpy()
                train_dir = osp.join(args.save_dir,"train")
                if not osp.exists(train_dir):
                    os.makedirs(train_dir)
                image_name = osp.join(train_dir,str(epoch)+"_"+str(iteration)+".png")
                save_image(sing_out,sing_gt,image_name)
           
        end_time = time.time()
        if rank==0:
            logger.info("epoch: {}, avg_loss: {:.5f}, time: {:.2f}s.\n".format(epoch,epoch_loss/(iteration+1),end_time-start_time))

        if rank==0 and (epoch % args.save_model_step) == 0:
            if not osp.exists(args.checkpoints):
                os.makedirs(args.checkpoints)
            if args.distributed:
                torch.save(network.module.state_dict(),osp.join(args.checkpoints,"epoch_"+str(epoch)+".pth")) 
            else:
                torch.save(network.state_dict(),osp.join(args.checkpoints,"epoch_"+str(epoch)+".pth")) 
        if rank==0 and args.test_flag:
            logger.info("epoch: {}, psnr and ssim test results:".format(epoch))
            if args.distributed:
                psnr_dict,ssim_dict = test(args,network.module,mask.cpu(),mask_s.cpu(),src_mask.cpu(),logger,writer=writer,epoch=epoch)
            else:
                psnr_dict,ssim_dict = test(args,network,mask.cpu(),mask_s.cpu(),src_mask.cpu(),logger,writer=writer,epoch=epoch)

            logger.info("psnr_dict: {}.".format(psnr_dict))
            logger.info("ssim_dict: {}.\n".format(ssim_dict))  
        

if __name__ == '__main__':
    args = parse_args()
    log_dir = osp.join(args.log_dir,"log")
    show_dir = osp.join(args.log_dir,"show")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(show_dir):
        os.makedirs(show_dir)
    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir = show_dir)
    
    rank = 0 
    if args.distributed:
        local_rank = int(args.local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    network = GAP_CCoT().to(args.device)

    if rank==0 and args.pre_train_model is not None:
        logger.info("Load pre_train model...")
        model_dict = network.state_dict()
        pretrained_dict = torch.load(args.pre_train_model)

        temp_dict = {}
        for k,v in pretrained_dict.items():
            # k = k.replace("cnn3d_net1","vit")
            if k in model_dict:
                temp_dict[k] = v
        # pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        pretrained_dict = temp_dict
        for k in pretrained_dict: 
            if model_dict[k].shape != pretrained_dict[k].shape:
                pretrained_dict[k] = model_dict[k]
                print("layer: {} parameters size is not same!".format(k))
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict,strict=False)
        # torch.save(network.vit.state_dict(),"vit.pth")
    else:            
        logger.info("No pre_train model")

    if args.distributed:
        network = DDP(network,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    mask, mask_s = generate_masks(args.mask_path,size_h=256,size_w=256)
    train(args,network,logger,mask,mask_s,writer)
    writer.close()
