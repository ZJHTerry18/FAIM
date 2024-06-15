import logging
from models.classifier import Classifier, NormalizedClassifier
from models.img_resnet import ResNet50, ResNet50CA, ResNet50CA_2
from models.img_resnet_var import ResNet50CA_var
from models.img_vit import ViT, ViT_pytorch, ViTCA_pytorch, ViTCA_pytorch_2, ViTCA_pytorch_3
from models.img_swint import SwinT, SwinTCA, SwinTCA_2
from models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50


__factory = {
    'resnet50': ResNet50,
    'resnet50ca': ResNet50CA,
    'resnet50cavar': ResNet50CA_var,
    'vit': ViT_pytorch,
    'vitca': ViTCA_pytorch_2,
    'swint': SwinT,
    'swintca': SwinTCA_2,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


def build_model(config, num_identities, num_clothes):
    logger = logging.getLogger('reid.model')
    # Build backbone
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        if config.MODEL.DECOUPLE:
            if config.MODEL.RELIABILITY:
                model_name = config.MODEL.NAME + 'cavar'
            else:
                model_name = config.MODEL.NAME + 'ca'
        else:
            model_name = config.MODEL.NAME
        logger.info("Init model: '{}'".format(model_name))
        kwargs = {'num_identities': num_identities, 'num_clothes': num_clothes}
        model = __factory[model_name](config, **kwargs)
    logger.info("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    # Build classifier
    if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth', 'generalcrossentropy']:
        identity_classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
        id_classifier = Classifier(feature_dim=config.MODEL.CA_DIM, num_classes=num_identities)
    else:
        identity_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_identities)
    
    if config.MODEL.DECOUPLE:
        clothes_classifier = Classifier(feature_dim=config.MODEL.CA_DIM, num_classes=num_clothes)
    else:
        clothes_classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_clothes)

    return model, identity_classifier, id_classifier, clothes_classifier