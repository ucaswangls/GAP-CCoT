import argparse 
import torch 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=2)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--log_dir",type=str,default="log")
    parser.add_argument("--save_model_step",type=int,default=1)
    parser.add_argument("--test_step",type=int,default=1)
    parser.add_argument("--save_dir",type=str,default="results")
    parser.add_argument("--save_image_step",type=int,default=100)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--show_step",type=int,default=60)
    parser.add_argument("--test_flag",type=bool,default=True)
    parser.add_argument("--checkpoints",type=str,default="checkpoints")
    parser.add_argument("--pre_train_model",type=str,default="checkpoints/last.pth")
    parser.add_argument("--train_data_path",type=str,default="/home/wanglishun/datasets/spectral/matlab")
    parser.add_argument("--test_data_path",type=str,default="/home/wanglishun/datasets/spectral/Testing_data")
    parser.add_argument("--mask_path",type=str,default="mask")
    parser.add_argument("--shift",type=bool,default=True)
    parser.add_argument("--real_data_path",type=str,default="/home/wanglishun/datasets/spectral/real_data/Testing_real_data")
    parser.add_argument("--real_mask_path",type=str,default="/home/wanglishun/datasets/spectral/real_data/")
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--local_rank",default=-1)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(args.local_rank) 
    if args.distributed:
        args.device = torch.device("cuda",local_rank)
    return args
