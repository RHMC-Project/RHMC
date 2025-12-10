import os
import argparse
import yaml
import torch
from torch import nn
from data import create_dataset
from utils import utils
from PIL import Image
from metrics.calculate_psnr_ssim import calculate_psnr_ssim
from metrics.calculate_lpips import calculate_lpips

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(config)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    config_path = args.config  
    config = load_config(config_path)
    opt = config.test
    dataset = create_dataset(opt)  

    ckpt_name = os.path.splitext(os.path.basename(opt.pretrain_model_path))[0]
   
    if opt.save_dir is None or opt.save_dir == '':
        assert opt.experiment_path is not None, "Please provide experiment path if save_dir is not provided"
        save_dir = os.path.join(opt.experiment_path,'result')
        save_dir = os.path.join(save_dir, ckpt_name)
        print(f"Saving images to default folder:{save_dir}")
        log_path = os.path.join(opt.experiment_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, f'{ckpt_name}_metrics.log')
    else:
        save_dir = opt.save_dir
        print(f"Saving images to custom folder:{save_dir}")
        log_path = os.path.join(save_dir, 'log')
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, f'{ckpt_name}_metrics.log')
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the model
    model_module = __import__(f"models.{opt.exp_name}.DA", fromlist=[''])
    print("Import model from:")
    print(model_module)
    generator_class = getattr(model_module, "DA")
    model_kwargs = vars(config.test.hyperparameters)
    network = generator_class(**model_kwargs)
    network = nn.DataParallel(network, device_ids=config.device_ids[0:1])
    model_path = os.path.join(opt.experiment_path,'ckpt',opt.pretrain_model_path)

    device = torch.device(f'cuda:{config.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print("Tesing on device:", device)

    # Load the pre-trained model
    state_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(state_dict)

    for data in dataset:
        inp = data['LR']
        with torch.no_grad():
            output_SR = network(inp)
        img_path = data['LR_paths']
        output_sr_img = utils.tensor_to_img(output_SR, normal=True)

        # Save the output image
        save_path = os.path.join(save_dir, img_path[0].split('/')[-1])
        save_img = Image.fromarray(output_sr_img)
        save_img.save(save_path)

    print(f"Images saved to {save_dir}")

    # Calculate metrics
    args = argparse.Namespace(test_dir=save_dir, gt_dir=opt.HRdataroot)
    psnr, ssim = calculate_psnr_ssim(args)
    lpips = calculate_lpips(args)
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}")

    # Log results

    with open(log_file, 'w') as f:
        f.write(f"Model: {ckpt_name}\n")
        f.write(f"PSNR: {psnr}\n")
        f.write(f"SSIM: {ssim}\n")
        f.write(f"LPIPS: {lpips}\n")

    print(f"Metrics saved to {log_file}")
