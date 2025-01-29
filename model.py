import os
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear
from torchvision.models import resnet18




class LASTED(nn.Module):
    def __init__(self, num_class=2, simclr_checkpoint_path=None):
        super().__init__()
        
        #Load CLIP model
        self.clip_model, self.preprocess = clip.load("RN50x64", device='cpu', jit=False)
        self.clip_model.eval()
        
        # Load SimCLR encoder
        self.simclr_encoder = self.load_simclr_encoder(simclr_checkpoint_path)
        self.simclr_encoder.eval()
        
        # Mapping layer from SimCLR features to CLIP's feature dimension (1024)
        self.simclr_mapping = nn.Linear(512, 1024) 
        
        # Combine CLIP and SimCLR features
        self.combined_mapping = nn.Linear(1024 + 1024, 1024)
        
        
        self.output_layer = nn.Sequential(
            nn.Linear(1024, 1280),
            nn.GELU(),
            nn.Linear(1280, 512),
        )
        self.fc = nn.Linear(512, num_class)
        self.text_input = clip.tokenize(['Real Map', 'Synthetic Map'])
        
    def load_simclr_encoder(self, checkpoint_path):
        # Define SimCLR encoder
        class SimCLR_Encoder(nn.Module):
            def __init__(self):
                super(SimCLR_Encoder, self).__init__()
                resnet = resnet18()
                # Remove the final fully connected layer
                self.encoder = nn.Sequential(*list(resnet.children())[:-1])

            def forward(self, x):
                h = self.encoder(x)
                h = h.squeeze()
                return h  # Return encoder output

        # Load the trained SimCLR encoder weights
        simclr_encoder = SimCLR_Encoder()
        if checkpoint_path is not None:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['model']

            # Extract encoder weights
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('f.'):
                    encoder_state_dict[k[len('f.'):]] = v 

            simclr_encoder.load_state_dict(encoder_state_dict, strict=False)

        return simclr_encoder
        
        
    def forward(self, image_input, isTrain=True):
        device = image_input.device
        
            
        simclr_image_features = self.simclr_encoder(image_input)
        
        simclr_image_features = simclr_image_features / simclr_image_features.norm(dim=1, keepdim=True)

        # Map SimCLR features to 1024 dimensions
        simclr_image_features_mapped = self.simclr_mapping(simclr_image_features)
        simclr_image_features_mapped = simclr_image_features_mapped / simclr_image_features_mapped.norm(dim=1, keepdim=True)

        # Combine CLIP and SimCLR features
        if isTrain:
            # During training, combine CLIP logits and SimCLR features
            image_feats = self.clip_model.encode_image(image_input)  # Extract features from CLIP
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        

            combined_features = torch.cat([image_feats, simclr_image_features_mapped], dim=-1)  
            
            combined_features = self.combined_mapping(combined_features)  
            combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)
            # Compute final logits using the fc layer
            final_features = self.output_layer(combined_features)
            logits = self.fc(final_features)
            return None, logits

        
        
        
        else:
            # During inference, combine CLIP and SimCLR features
            image_feats = self.clip_model.encode_image(image_input)  # Extract features from CLIP
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            combined_features = torch.cat([image_feats, simclr_image_features_mapped], dim=-1)  # [batch_size, 2048]
            combined_features = self.combined_mapping(combined_features)  # Map back to [batch_size, 1024]
            combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)
        # Return combined features
            return None, combined_features
        
        
        


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    simclr_checkpoint_path = './SIMCLR/Weights/best_simclr_checkpoint.ckpt'

    model = LASTED(simclr_checkpoint_path=simclr_checkpoint_path).cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params: %.2f' % (params / (1024 ** 2)))

    x = torch.zeros([4, 3, 448,448]).cuda()
    _, logits = model(x)
    print("Shape of logits returned by model:", logits.shape)
