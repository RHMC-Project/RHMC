import os
import datetime
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import pywt
import numpy as np
from utils.utils import tensor_to_img
from data import create_dataset
from losses.WaveletLoss import WaveletLoss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



parser = argparse.ArgumentParser(description="Two-Stage Training Script")
parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)
            

def print_config(config, indent=0):
    for key, value in config.__dict__.items():
        prefix = ' ' * indent
        if isinstance(value, Config):
            print(f"{prefix}{key}:")
            print_config(value, indent + 2)
        else:
            print(f"{prefix}{key}: {value}")
            

def create_exp_dir(root_path='./experiments', exp_name=None):
    if exp_name is None:
        exp_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        exp_name += datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(root_path, exp_name)
    models_path = os.path.join(exp_path, 'models') 
    ckpt_path = os.path.join(exp_path, 'ckpt') 
    result_path = os.path.join(exp_path, 'result') 
    log_path = os.path.join(exp_path, 'log') 
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    return exp_path, models_path, ckpt_path, result_path, log_path


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(config)


def copy_code_files(src, dst):
    if os.path.exists(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        print(f"Source path {src} does not exist.")
        raise FileNotFoundError



if __name__ == '__main__':
    # Load configuration
    config_path = args.config
    config = load_config(config_path)

    device_ids = config.device_ids
    opt = config.train
    
    
    torch.cuda.set_device(device_ids[0])

    # Create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    iters_perepoch = dataset_size // opt.batch_size
    
    print('The number of training images = %d' % dataset_size)
    single_epoch_iters = (dataset_size // opt.batch_size)
    
    #start_iter = opt.resume_iter
    
   
    stage1_epochs = getattr(opt, 'stage1_epochs', 200)  
    stage2_epochs = getattr(opt, 'stage2_epochs', 50)  
    total_epochs = stage1_epochs + stage2_epochs
    total_iters = opt.total_epochs * single_epoch_iters
    
    print(f"Stage 1 training epochs: {stage1_epochs}")
    print(f"Stage 2 training epochs: {stage2_epochs}")
    print(f"Total training epochs: {total_epochs}")

    # Create validation dataset
    val_dataset = create_dataset(opt)
    #print_config(val_dataset, indent=0)

    # Load model dynamically
    model_module = __import__(f"model.{opt.exp_name}.HMC", fromlist=[''])
    print("Import model from:")
    print(model_module)
    generator_class = getattr(model_module, "two_stageHMC")
    model_kwargs = vars(config.train.hyperparameters)
    print("Model kwargs:")
    print(model_kwargs)
    generator = generator_class(**model_kwargs)
    generator = nn.DataParallel(generator, device_ids=device_ids)
    generator = generator.cuda(device=device_ids[0])
    


    
    criterionL1 = nn.L1Loss()
    
    
    exp_path, models_path, ckpt_path, result_path, log_path = create_exp_dir(exp_name=opt.exp_name)

   
    codepath = os.path.join(opt.codepath, opt.exp_name)
    copy_code_files(codepath, models_path)

    
    with open(os.path.join(exp_path, 'config.yaml'), 'w') as file: 
        yaml.dump(config.__dict__, file) 

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=log_path)


  
    generator.train()
    
    
    optimizer_G_stage1 = optim.AdamW(generator.parameters(), lr=opt.lr, 
                                     betas=(opt.beta1, opt.beta2), weight_decay=0.02)
    scheduler_G_stage1 = lr_scheduler.MultiStepLR(optimizer_G_stage1, 
                                                  milestones=[50 , 100 ,150 ,180], gamma=0.5)
    cur_iters = 0
    
    
    for cur_epoch in range(1, stage1_epochs + 1):
        epoch_loss = 0
        generator.train()

        for i, data in enumerate(dataset):
            cur_iters += 1
            hr = data['HR'].cuda(device=device_ids[0])
            lr = data['LR'].cuda(device=device_ids[0])
            
            output = generator(lr)
            loss = criterionL1(hr, output)
            
            optimizer_G_stage1.zero_grad()
            loss.backward()
            optimizer_G_stage1.step()

            epoch_loss += loss.item()
            
            if cur_iters % opt.print_freq == 0:
                print("Stage1 - Epoch:[%d/%d] Iter:[%d] Loss_L1: %.6f" % 
                      (cur_epoch, stage1_epochs, cur_iters, loss.item()))
            
            if cur_iters % opt.save_latest_freq == 0:
                print("saving latest ckpt")
                torch.save(generator.state_dict(), os.path.join(ckpt_path, 'latest1_model.pt'))
        
    
        
        if cur_epoch % opt.save_epoch_freq == 0:
            torch.save(generator.state_dict(), 
                      os.path.join(ckpt_path, f'stage1_epoch{cur_epoch:03d}_iter{cur_iters:03d}.pt'))
        print(f"Stage1 model saved: stage1_epoch{cur_epoch:03d}_iter{cur_iters:03d}.pt")        
        
        
        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataset:
                hr = data['HR'].cuda(device=device_ids[0])
                lr = data['LR'].cuda(device=device_ids[0])
                output = generator(lr)
                val_loss += criterionL1(hr, output).item()

        val_loss /= len(val_dataset)
        
       
        writer.add_scalar('Stage_Loss/train', epoch_loss / len(dataset), cur_epoch)
        writer.add_scalar('Stage_Loss/val', val_loss, cur_epoch)
        
        # Log generated images
        if cur_epoch % opt.save_epoch_freq  == 0:
            # output_image = output[0].cpu().detach().numpy().transpose(1, 2, 0)
            output_image = tensor_to_img(output[0], normal=True)
            writer.add_image('stage1-Generated Image', output_image, cur_epoch, dataformats='HWC')

        

        scheduler_G_stage1.step()
        lr_G = scheduler_G_stage1.get_lr()
        print("current learning rate is:")
        print(lr_G)
        print(f"Stage - Epoch {cur_epoch} completed, Val Loss: {val_loss:.6f}")

    
    torch.save(generator.state_dict(), os.path.join(ckpt_path, 'stage_final.pt'))
   

    
    writer.close()