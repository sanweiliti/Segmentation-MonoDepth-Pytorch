import yaml
import torch
import argparse

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScoreDepth, averageMeter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(cfg, args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'], cfg['task'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['test_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        img_norm=cfg['data']['img_norm']
    )

    n_classes = 0
    running_metrics_val = runningScoreDepth(cfg['data']['dataset'])

    testloader = data.DataLoader(loader,
                                 batch_size=cfg['training']['batch_size'],
                                 num_workers=0)

    # Load Model
    model = get_model(cfg['model'], cfg['task'], n_classes=n_classes).to(device)
    #weights = torch.load(cfg['testing']['trained_model'])
    weights = torch.load(cfg['testing']['trained_model'], map_location=lambda storage, loc: storage)
    model.load_state_dict(weights["model_state"])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (images, labels, img_path) in tqdm(enumerate(testloader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # [batch_size, n_classes, height, width]
            if cfg['model']['arch'] == "dispnet" and cfg['task'] == "depth":
                outputs = 1 / outputs

            pred = outputs.squeeze(1).data.cpu().numpy()
            gt = labels.data.squeeze(1).cpu().numpy()

            running_metrics_val.update(gt=gt, pred=pred)

    val_result = running_metrics_val.get_scores()
    for k, v in val_result.items():
        print(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn_cityscapes_depth.yml",
        help="Config file to be used",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    test(cfg, args)

