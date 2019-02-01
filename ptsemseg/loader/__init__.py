from ptsemseg.loader.cityscapes_loader_seg import cityscapesLoader_seg
from ptsemseg.loader.kitti_loader_seg import kittiLoader_seg
from ptsemseg.loader.kitti_loader_depth import kittiLoader_depth


def get_loader(name, task):
    if task == "seg":
        return {
            "cityscapes": cityscapesLoader_seg,
            "kitti": kittiLoader_seg
        }[name]
    elif task == "depth":
        return {
            "kitti": kittiLoader_depth
        }[name]
    else:
        print("task undefined!")
