import copy
import torchvision.models as models
from collections import OrderedDict

from ptsemseg.models.fcn_seg import *
from ptsemseg.models.segnet_seg import *
from ptsemseg.models.frrn_seg import *
from ptsemseg.models.deeplab_seg import *
from ptsemseg.models.fcrn_seg import *
from ptsemseg.models.dispnet_seg import *

from ptsemseg.models.fcn_depth import *
from ptsemseg.models.segnet_depth import *
from ptsemseg.models.frrn_depth import *
from ptsemseg.models.deeplab_depth import *
from ptsemseg.models.fcrn_depth import *
from ptsemseg.models.dispnet_depth import *


def get_model(model_dict, task, n_classes):
    name = model_dict['arch']
    model = _get_model_instance(name, task)  # model: an instance of class fcn8s
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if task == "seg":
        model = model(n_classes=n_classes, **param_dict)
    elif task == "depth":
        model = model(**param_dict)

    if name == "frrn":
        pass

    elif name == "fcn":
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

        # if you want to load from downloaded pretrained model:
        # vgg16 = models.vgg16(pretrained=False)
        # vgg16.load_state_dict(torch.load("pretrained_models/vgg16-imagenet.pth"))
        # model.init_vgg16_params(vgg16)

    elif name == "segnet":
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

        # if you want to load from downloaded pretrained model:
        # vgg16 = models.vgg16(pretrained=False)
        # vgg16.load_state_dict(torch.load("pretrained_models/vgg16-imagenet.pth"))
        # model.init_vgg16_params(vgg16)

    elif name == "dispnet":
        model.init_weights()

    elif name == "deeplab":
        resnet101 = models.resnet101(pretrained=True)
        initial_state_dict = model.init_resnet101_params(resnet101)
        model.load_state_dict(initial_state_dict, strict=False)

        # if you want to load from downloaded pretrained model:
        # model_path = 'pretrained_models/resnet101-imagenet.pth'
        # new_state_dict = model.init_resnet101_params(model_path)
        # model.load_state_dict(new_state_dict, strict=False)

    elif name == "fcrn":
        resnet50 = models.resnet50(pretrained=True)
        init_state_dict = model.init_resnet50_params(resnet50)
        model.load_state_dict(init_state_dict, strict=False)

        # if you want to load from downloaded pretrained model:
        # model_path = 'pretrained_models/resnet50-imagenet.pth'
        # init_state_dict = model.init_resnet50_params(model_path)
        # model.load_state_dict(init_state_dict, strict=False)

    else:
        print("Model {} not available".format(name))

    return model


def _get_model_instance(name, task):
    try:
        if task == "seg":
            return {
                "fcn": fcn_seg,
                "segnet": segnet_seg,
                "frrn": frrn_seg,
                "dispnet": dispnet_seg,
                "deeplab": deeplab_seg,
                "fcrn": fcrn_seg,
            }[name]
        elif task == "depth":
            return {
                "fcn": fcn_depth,
                "segnet": segnet_depth,
                "frrn": frrn_depth,
                "dispnet": dispnet_depth,
                "deeplab": deeplab_depth,
                "fcrn": fcrn_depth,
            }[name]
    except:
        raise("Model {} not available".format(name))
