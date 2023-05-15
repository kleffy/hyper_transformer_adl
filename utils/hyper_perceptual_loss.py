import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms

class HyperPerceptualLoss(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', pretrained=True, 
                 in_channels=224, out_features=4096, saved_model_path=None,
                 mean_file=None, std_file=None, normalize=False, perceptual_model=None
                 ):
        super(HyperPerceptualLoss, self).__init__()
        
        # Load the pretrained efficientnetv2 model
        self.saved_model_path = saved_model_path
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.out_features = out_features
        self.mean_file = mean_file
        self.std_file = std_file
        self.normalize = normalize
        self.perceptual_model = perceptual_model
        
        if mean_file:
            self.mean = np.load(mean_file)
        
        if std_file:
            self.std = np.load(std_file)
        
        if perceptual_model:
            self.model = self.perceptual_model
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_channels, num_classes=out_features)
            num_in_features = self.model.get_classifier().in_features
            self.model.fc = nn.Sequential(
                        nn.Linear(in_features=num_in_features, out_features=num_in_features),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(num_in_features),
                        nn.Dropout(p=0.4),
                        nn.Linear(in_features=num_in_features, out_features=out_features),
                    )
        
        checkpoint = torch.load(self.saved_model_path)
        self.model.load_state_dict(checkpoint, strict=False)
        # print(self.model)
        # Freeze all the parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def forward(self, input_image, target_image):
        input_image = self._process_image(input_image)
        # target_image = self._process_image(target_image)
        
        # Get the feature maps from the pretrained model for the input and target images
        with torch.no_grad():
            if self.perceptual_model:
                input_features = self.model(input_image.unsqueeze(2))
                target_features = self.model(target_image.unsqueeze(2))
            else:
                input_features = self.model(input_image)
                target_features = self.model(target_image)
        
        loss = F.mse_loss(input_features, target_features, reduction='mean')
        
        return loss
    
    def _process_image(self, image):

        if self.normalize:
            normalize_transform = transforms.Normalize(mean=self.mean, std=self.std + 1e-8)
            image = normalize_transform(image)

        # image = self._normalize_hyperspectral_image(image)
        
        return image
    
    def _normalize_hyperspectral_image(self, image):
        print(f"b4 unsq: {image.shape}")
        # if self.perceptual_model:
        #     image = image.unsqueeze(2)
        #     print(f"after unsq: {image.shape}")
        min_vals = image.view(image.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = image.view(image.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        eps =  1e-8 #torch.tensor([1e-8], dtype=torch.float32)
        normalized_image = (image - min_vals) / (max_vals - min_vals + eps)
        # if self.perceptual_model:
        #     normalized_image = normalized_image.squeeze(2)
        #     print(f"after sq normalized_image: {normalized_image.shape}")
        return normalized_image
