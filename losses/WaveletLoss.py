import torch
import pywt
import  torch.nn as nn

class WaveletLoss(nn.Module):
   
    def __init__(self, wavelet='db1', alpha=1.0, beta1=1.0, beta2=1.0, beta3=1.0):
        super(WaveletLoss, self).__init__()
        self.wavelet = wavelet
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.l1_loss = nn.L1Loss()
    
    def dwt_2d(self, x):
        
        batch_size, channels, height, width = x.shape
        
       
        LL = torch.zeros(batch_size, channels, height//2, width//2, device=x.device)
        LH = torch.zeros(batch_size, channels, height//2, width//2, device=x.device)
        HL = torch.zeros(batch_size, channels, height//2, width//2, device=x.device)
        HH = torch.zeros(batch_size, channels, height//2, width//2, device=x.device)
        
        for b in range(batch_size):
            for c in range(channels):
                
                img = x[b, c].cpu().numpy()
                coeffs = pywt.dwt2(img, self.wavelet)
                cA, (cH, cV, cD) = coeffs
                
                
                LL[b, c] = torch.from_numpy(cA).to(x.device)
                LH[b, c] = torch.from_numpy(cH).to(x.device)
                HL[b, c] = torch.from_numpy(cV).to(x.device)
                HH[b, c] = torch.from_numpy(cD).to(x.device)
        
        return LL, LH, HL, HH
    
    def forward(self, sr_img, gt_img):
        
        LL_sr, LH_sr, HL_sr, HH_sr = self.dwt_2d(sr_img)
        LL_gt, LH_gt, HL_gt, HH_gt = self.dwt_2d(gt_img)
       
        loss_LL = self.l1_loss(LL_sr, LL_gt)
        loss_LH = self.l1_loss(LH_sr, LH_gt)
        loss_HL = self.l1_loss(HL_sr, HL_gt)
        loss_HH = self.l1_loss(HH_sr, HH_gt)
        
       
        total_loss = (self.alpha * loss_LL + 
                     self.beta1 * loss_LH + 
                     self.beta2 * loss_HL + 
                     self.beta3 * loss_HH)
        
        return total_loss