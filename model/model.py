import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# BREM
class BoundaryRefinementAndEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryRefinementAndEnhancementModule, self).__init__()
        
        self.wavelet_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.wavelet_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.wavelet_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        
        
        self.guided_filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )
        
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        
        edges1 = self.wavelet_conv1(x)
        edges2 = self.wavelet_conv2(x)
        edges3 = self.wavelet_conv3(x)
        
       
        edges = edges1 + edges2 + edges3
        
        
        refined_edges = self.guided_filter(edges)
        
        
        refined_edges = self.dropout(refined_edges)
        
        
        return x + refined_edges

# LCAD
class LightweightChannelAttentiveDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super(LightweightChannelAttentiveDecoder, self).__init__()

        
        self.attention1 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[-1] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(encoder_channels[-1] // 4, encoder_channels[-1], kernel_size=1),
            nn.Sigmoid()
        )
        self.upconv1 = nn.ConvTranspose2d(encoder_channels[-1], encoder_channels[-2], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(encoder_channels[-2], encoder_channels[-2], kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.3)  

        
        self.attention2 = nn.Sequential(
            nn.Conv2d(encoder_channels[-2], encoder_channels[-2] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(encoder_channels[-2] // 4, encoder_channels[-2], kernel_size=1),
            nn.Sigmoid()
        )
        self.upconv2 = nn.ConvTranspose2d(encoder_channels[-2], encoder_channels[-3], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(encoder_channels[-3], encoder_channels[-3], kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.3)

        
        self.attention3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-3], encoder_channels[-3] // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(encoder_channels[-3] // 4, encoder_channels[-3], kernel_size=1),
            nn.Sigmoid()
        )
        self.upconv3 = nn.ConvTranspose2d(encoder_channels[-3], encoder_channels[-4], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(encoder_channels[-4], encoder_channels[-4], kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.3)

        
        self.final_conv = nn.Conv2d(encoder_channels[-4], 1, kernel_size=1)

    def forward(self, encoder_features):
        
        x = encoder_features[-1]

        
        x = x * self.attention1(x)
        x = self.upconv1(x)
        x = F.interpolate(x, size=encoder_features[-2].shape[2:], mode='bilinear', align_corners=False)
        x = x + encoder_features[-2]  
        x = self.conv1(x)
        x = self.dropout1(x)

        
        x = x * self.attention2(x)
        x = self.upconv2(x)
        x = F.interpolate(x, size=encoder_features[-3].shape[2:], mode='bilinear', align_corners=False)
        x = x + encoder_features[-3]  
        x = self.conv2(x)
        x = self.dropout2(x)

        
        x = x * self.attention3(x)
        x = self.upconv3(x)
        x = F.interpolate(x, size=encoder_features[-4].shape[2:], mode='bilinear', align_corners=False)
        x = x + encoder_features[-4] 
        x = self.conv3(x)
        x = self.dropout3(x)

        
        x = self.final_conv(x)

        
        x = F.interpolate(x, size=encoder_features[0].shape[2:], mode='bilinear', align_corners=False)

        return x

# BRLAN with EfficientNet-b5 backbone, BREM and LCAD
class MedLightNet(nn.Module):
    def __init__(self, n_channels=3):
        super(BRLAN, self).__init__()

        
        self.encoder = smp.encoders.get_encoder(
            "timm-efficientnet-b5",
            in_channels=n_channels,
            depth=5,
            weights="noisy-student"
        )

        
        self.brem = BoundaryRefinementAndEnhancementModule(self.encoder.out_channels[-1])

        
        self.lcad = LightweightChannelAttentiveDecoder(self.encoder.out_channels)

    def forward(self, x):
        
        encoder_features = self.encoder(x)
        encoder_output = encoder_features[-1]

        
        refined_features = self.brem(encoder_output)

        
        encoder_features[-1] = refined_features

        
        output = self.lcad(encoder_features)

        return output

def get_model():
    return BRLAN()


if __name__ == "__main__":
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    print(model)

    x = torch.randn((1, 3, 192, 192)).to(device)

    try:
        outputs = model(x)
        print(outputs.shape)  
    except RuntimeError as e:
        print("RuntimeError:", e)

    batch_size = 4
    data = torch.randn((batch_size, 3, 192, 192)).to(device)
    target = torch.randint(0, 2, (batch_size, 192, 192)).to(device)

    try:
        outputs = model(data)
        print("Model output shape:", outputs.shape)

        
        upsampled_logits = F.interpolate(outputs, size=target.shape[-2:], mode="bilinear", align_corners=False)
        print("Upsampled logits shape:", upsampled_logits.shape)
    except RuntimeError as e:
        print("RuntimeError during forward pass:", e)

    gc.collect()
    torch.cuda.empty_cache()
