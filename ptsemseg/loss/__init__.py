import copy
import logging
import functools

from ptsemseg.loss.loss import cross_entropy2d
from ptsemseg.loss.loss import bootstrapped_cross_entropy2d

from ptsemseg.loss.loss import l1_loss
from ptsemseg.loss.loss import Berhu_loss
from ptsemseg.loss.loss import Huber_loss
from ptsemseg.loss.loss import scale_invariant_loss

logger = logging.getLogger('ptsemseg')

key2loss = {'cross_entropy': cross_entropy2d,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'l1_loss': l1_loss,
            'berhu_loss': Berhu_loss,
            'huber_loss': Huber_loss,
            'scale_invariant_loss': scale_invariant_loss}

def get_loss_function(cfg):
    if cfg['training']['loss'] is None:
        if cfg['task'] == "seg":
            logger.info("Using default cross entropy loss for segmentation")
            return cross_entropy2d
        elif cfg['task'] == "depth":
            logger.info("Using default scale invariant loss for depth")
            return scale_invariant_loss
        else:
            print("Please specify the loss!")

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name, 
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
