import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import util as u
from models import resnet, alexnet, densenet, shufflenet, mobilenet, resnext
from typing import Optional, Union


# Base functionality
class UtilBase(nn.Module):
    def __init__(self):
        super(UtilBase, self).__init__()
        self.config = config.get_global_config()
        self.classifier: Optional[Union[nn.Sequential, nn.Module]] = None
        self.feature_extractor: Optional[Union[nn.Sequential, nn.Module]] = None
        self.classifier_fc_layers = nn.ModuleList()

    def get_parameters(self):
        c = config.get_global_config()
        params = list(y for x, y in self.named_parameters() if x.split('.')[1] not in c.ignore_params) if c.ignore_params else self.parameters()
        ignored_params = list(x for x, y in self.named_parameters() if x.split('.')[1] in c.ignore_params)
        for name, tensor in self.named_parameters():
            if name in ignored_params:
                tensor.requires_grad = False
            else:
                tensor.requires_grad = True
        print("Ignoring parameters: ", ignored_params)
        return params

    def init_classifier(self, emb_size, num_classes):
        self.classifier_fc_layers.append(nn.Linear(emb_size, emb_size, bias=True))
        self.classifier_fc_layers.append(nn.Linear(emb_size, num_classes, bias=True))
        nn.init.eye_(self.classifier_fc_layers[0].weight)
        nn.init.constant_(self.classifier_fc_layers[0].bias, 0)

    def classify(self, x, dropout: float = 0.):
        x = F.dropout(x, p=dropout, training=True)
        x = self.classifier_fc_layers[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.classifier_fc_layers[1](x)
        return x

    def forward(self, img, phase):
        mc_buffer = []
        pre_drop_features = self.feature_extractor(img)
        if phase in [u.Phase(u.Phase.ADAPTATION_TRAIN), u.Phase(u.Phase.SOURCE_ONLY_TRAIN)]:
            logits = self.classify(pre_drop_features, dropout=0.50)
        elif phase in [u.Phase(u.Phase.MC_DROPOUT)]:
            for i in range(self.config.uncertain_mc_iters):
                mc_buffer.append(self.classify(pre_drop_features, dropout=self.config.uncertain_mc_dropout).unsqueeze(1).detach())
            mc_buffer = torch.cat(mc_buffer, dim=1)
            logits = self.classify(pre_drop_features, dropout=self.config.uncertain_mc_dropout)
        else:
            new_features = pre_drop_features
            logits = self.classify(new_features, dropout=0.)

        return {'logits': logits,
                'features': pre_drop_features,
                'mc_logits': mc_buffer,
                }


################################################################################################################################################################
# Available networks
class Resnet50(UtilBase):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet.resnet50(pretrained=self.config.pretrained)
        self.embedding_size = self.feature_extractor.embedding_size
        self.init_classifier(self.embedding_size, self.config.current.num_classes)


class Resnet101(UtilBase):
    def __init__(self):
        super().__init__()
        self.feature_extractor = resnet.resnet101(pretrained=self.config.pretrained)
        self.embedding_size = self.feature_extractor.embedding_size
        self.init_classifier(self.embedding_size, self.config.current.num_classes)


class Densenet121(UtilBase):
    def __init__(self):
        super().__init__()
        self.feature_extractor = densenet.densenet121(pretrained=self.config.pretrained)
        self.embedding_size = self.feature_extractor.embedding_size
        self.init_classifier(self.embedding_size, self.config.current.num_classes)


class Mobilenet_v2(UtilBase):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet.mobilenet_v2(pretrained=self.config.pretrained)
        self.embedding_size = self.feature_extractor.embedding_size
        self.init_classifier(self.embedding_size, self.config.current.num_classes)


################################################################################################################################################################
# Util functions for listing and getting available models
def get_model(model_name, gpus):
    # Look for all defined subclasses of nn.Module
    defined_models = {v.__name__: v for k, v in globals().items() if type(v) == type(type) and issubclass(v, nn.Module)}
    chosen_model = defined_models[model_name]()
    chosen_model = nn.DataParallel(chosen_model, device_ids=gpus).cuda()
    # Calculate parameter count
    num_params, num_backbone_params = 0, 0
    for p in chosen_model.parameters():
        num_params += p.numel()
    for p in chosen_model.module.feature_extractor.parameters():
        num_backbone_params += p.numel()
    print(f"Loaded model {chosen_model.module.__class__.__name__} with {num_params:,} parameters (backbbone {num_backbone_params:,})")
    return chosen_model


def get_available():
    return [v.__name__ for k, v in globals().items() if type(v) == type(type) and issubclass(v, nn.Module)]


def get_model_by_name(name):
    return {v.__name__.lower(): v for k, v in globals().items() if type(v) == type(type) and
            issubclass(v, nn.Module) and
            v.__name__.lower() != UtilBase.__name__.lower()}[name.lower()]
