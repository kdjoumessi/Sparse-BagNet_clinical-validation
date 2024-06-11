import torch
import torch.nn as nn
from torchvision import models
import bagnets.pytorchnet

from utils.func import print_msg, select_out_features


def generate_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    model = build_model(
        cfg,
        cfg.train.network,
        out_features,
        cfg.train.pretrained)

    if cfg.train.checkpoint: # load the weight of sparse BagNet with SA
        weights = torch.load(f"./model/{cfg.train.checkpoint}")
        model.load_state_dict(weights, strict=True)
        print_msg('Checkpoint. Load weights form {}'.format(cfg.train.checkpoint))
    
    if cfg.train.pretrained_midl:
        path = f"./model/{cfg.train.pretrained_midl}"
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new model state dict
        model.load_state_dict(model_dict)
        print('MIDL pretrained: ', path)
    
    if cfg.base.device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(cfg.base.device)
    return model

def build_model(cfg, network, num_classes, pretrained=False):
    model = BUILDER[network](pretrained=pretrained)       

    if 'resnet' in network or 'resnext' in network or 'shufflenet' in network:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'bagnet' in network:       
        if cfg.train.version == 'v1':
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else: # nn.Sequential(*list(model.children())[:-2]) 
            new_model =  list(model.children())[:-2]            
            model = Bagnet_v2(cfg, new_model, num_classes)
    elif 'densenet' in network:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'vgg' in network:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    elif 'mobilenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    elif 'squeezenet' in network:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'bagnet33': bagnets.pytorchnet.bagnet33,
}


class Bagnet_v2(nn.Module):
    def __init__(self, cfg, model, num_classes, n=60, m=60, num_fmap_channels=2048, att_dim=2048):
        super(Bagnet_v2, self).__init__()
        self.cfg=cfg
        self.k_min = cfg.train.k_min
        self.num_classes = num_classes

        self.sequential = nn.Sequential(*model)
        # classification layer: conv2d instead of FCL
        self.conv2 = nn.Conv2d(2048, num_classes, kernel_size=(1,1), stride=1)

        # Avg pool without softmax => average value within each heatmap in their respective class
        self.avgpool = nn.AvgPool2d(kernel_size=(n, m), stride=(1,1), padding=0)

        # patch extractor for the attention module
        #self.patch_extraction = nn.AvgPool2d(kernel_size=(1,1), stride=1)

        # attention module
        self.att_tanh = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Tanh()
        )
        self.att_sigm = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Sigmoid()
        )
        self.att_outer = nn.Sequential(
            nn.Linear(num_fmap_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sequential(x)           # bs, C, h, w ~ (60, 60)
        #bs, C, n, m = x.shape           # spatial shape (h, w) = (60, 60) 
        x_local = self.conv2(x)          # bs, n_class, h, w => classification layer: conv2d instead of FCL

        #xx = self.avgpool(x_local)       # bs, n_class, 1, 1 ~ SAP: not useful
        #out = xx.view(xx.shape[0], -1)  # bs, n_class

        # patch extractor for the attention module
        #xxx = self.patch_extraction(x)   # bs, C   
        #b, c, _, _ = x.shape
        # create single patches: bs, patch, C
        x = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1)
        b, k, c = x.shape
        x = x.reshape(-1, c) # bs*n_patches, C

        x_local  = x_local.view(b, k, self.num_classes)
        x_weight = self.att_outer(self.att_tanh(x) * self.att_sigm(x)).view(b,k,1)
        
        pred = torch.sum(x_local * x_weight, dim=1) #/ torch.clamp(torch.sum(x_weight), min=self.k_min)

        return pred, x_local, x_weight     