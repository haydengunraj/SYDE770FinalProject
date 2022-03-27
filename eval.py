import torch
import numpy as np

from metrics import OverlayLogMetric


def intersection_union(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    intersection = torch.sum(y_true*y_pred)
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection, union


def evaluate_segmentation(model, data_loader, writer=None, step=None, apply_sigmoid=False, num_vis=4, device='cuda:0'):
    model.eval()
    intersection = 0
    union = 0
    true_metric = OverlayLogMetric('seg_true', 0, 'image', 'mask')
    pred_metric = OverlayLogMetric('seg_pred', 0, 'image', 'mask')
    vis_batches = set(np.random.choice(list(range(len(data_loader))), num_vis))
    with torch.no_grad():
        for k, batch in enumerate(data_loader):
            image, mask = (t.to(device) for t in batch)
            pred_mask = model(image)
            if apply_sigmoid:
                pred_mask = torch.sigmoid(pred_mask)
            i, u = intersection_union(mask, pred_mask)
            intersection += i
            union += u

            if k in vis_batches:
                true_metric.update({'image': image, 'mask': mask})
                pred_metric.update({'image': image, 'mask': pred_mask})

    iou = intersection/union
    dsc = 2*intersection/(union + intersection)

    if writer is not None:
        true_metric.log(writer, step)
        pred_metric.log(writer, step)
        writer.add_scalar('val/IoU', iou, step)
        writer.add_scalar('val/DSC', dsc, step)
        writer.flush()

    return iou, dsc


if __name__ == '__main__':
    import argparse
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    from transforms import ExteriorExclusion
    from datasets import SegmentationDataset
    from model import UNet

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='Path to model checkpoint')
    parser.add_argument('-dd', '--data_dir', type=str,
                        default='F:\\Datasets\\COVID-19-20\\COVID-19-20_v2\\segmentation_datasets\\validation',
                        help='Folder containing images')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-dv', '--device', type=str, default='cuda:0', help='Device code')
    parser.add_argument('-sz', '--size', type=int, default=256, help='Square image size')
    args = parser.parse_args()

    # Make model
    seg_model = UNet()
    seg_model.to(args.device)

    # Load weights
    weights = torch.load(args.ckpt)
    seg_model.load_state_dict(weights['seg_model'])

    # Make dataset
    val_tform = T.Compose([
        ExteriorExclusion(),
        T.Resize([args.size, args.size]),
        T.ToTensor()
    ])
    seg_tform = T.Compose([
        T.Resize([args.size, args.size]),
        T.ToTensor()
    ])
    val_dataset = SegmentationDataset(args.data_dir, transform=val_tform, target_transform=seg_tform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # Run evaluation
    iou, dsc = evaluate_segmentation(seg_model, val_loader, device=args.device)
    print('Evaluation Results\n' + '='*18)
    print('IoU: {:.4f}'.format(iou))
    print('DSC: {:.4f}'.format(dsc))
