import os
import random
import csv
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
from pycocotools.coco import COCO
from pycocotools.mask import encode as mask_encode
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())
        self.cat_id_to_label = {1: 1}  # single category Garbage=1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []
        for ann in annotations:
            if ann['category_id'] != 1:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x+w, y+h])
            labels.append(self.cat_id_to_label[ann['category_id']])
            m = self.coco.annToMask(ann)
            if m.sum() == 0:
                continue
            masks.append(m)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
        else:
            masks = torch.zeros((0, image.size[1], image.size[0]), dtype=torch.uint8)

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

def get_transforms():
    return transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, data_loader, device, epoch, val_images_dir):
    model.eval()
    coco = data_loader.dataset.coco
    img_ids = data_loader.dataset.image_ids

    results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", leave=False):
            images = list(img.to(device) for img in images)
            image_ids = [t["image_id"].item() for t in targets]

            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = image_ids[i]
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                masks = output["masks"].cpu().numpy()

                for j, score in enumerate(scores):
                    if score < 0.05:
                        continue
                    m = (masks[j, 0] > 0.5).astype(np.uint8)
                    rle = mask_encode(np.asfortranarray(m))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    cat_id = 1
                    x_min, y_min, x_max, y_max = boxes[j]
                    res = {
                        "image_id": image_id,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "score": float(score),
                        "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                    }
                    results.append(res)

    if len(results) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cocores = coco.loadRes(results)

    # Bounding Box Evaluation
    coco_eval_bbox = COCOeval(coco, cocores, "bbox")
    coco_eval_bbox.params.imgIds = img_ids
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()

    map_bbox_50_95 = coco_eval_bbox.stats[0]  # AP (IoU=0.5:0.95)
    map_bbox_50 = coco_eval_bbox.stats[1]    # AP (IoU=0.5)

    # Compute AP@0.95 for bounding boxes
    precision_bbox = coco_eval_bbox.eval['precision']
    iou_thrs = coco_eval_bbox.params.iouThrs
    iou_95_index = np.where(iou_thrs == 0.95)[0][0]
    p_95_bbox = precision_bbox[iou_95_index]
    p_95_bbox = p_95_bbox[p_95_bbox > -1]
    map_bbox_95 = np.mean(p_95_bbox) if p_95_bbox.size > 0 else 0.0

    # Segmentation (Mask) Evaluation
    coco_eval_mask = COCOeval(coco, cocores, "segm")
    coco_eval_mask.params.imgIds = img_ids
    coco_eval_mask.evaluate()
    coco_eval_mask.accumulate()
    coco_eval_mask.summarize()

    map_mask_50_95 = coco_eval_mask.stats[0]  # AP (IoU=0.5:0.95)
    map_mask_50 = coco_eval_mask.stats[1]    # AP (IoU=0.5)

    # Compute AP@0.95 for masks
    precision_mask = coco_eval_mask.eval['precision']
    p_95_mask = precision_mask[iou_95_index]
    p_95_mask = p_95_mask[p_95_mask > -1]
    map_mask_95 = np.mean(p_95_mask) if p_95_mask.size > 0 else 0.0

    return map_bbox_50_95, map_bbox_50, map_bbox_95, map_mask_50_95, map_mask_50, map_mask_95

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_annotation = "coco_training_annotations.json"
    train_images_dir = "train/images"
    val_annotation = "coco_validation_annotations.json"
    val_images_dir = "val/images"

    train_dataset = COCODataset(train_annotation, train_images_dir, transforms=get_transforms())
    val_dataset = COCODataset(val_annotation, val_images_dir, transforms=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 15
    best_map = -1.0
    csv_file = "training_metrics.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "bbox_mAP_0.5_0.95", "bbox_mAP_0.5", "bbox_mAP_0.95", 
                         "mask_mAP_0.5_0.95", "mask_mAP_0.5", "mask_mAP_0.95"])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()

        train_loss = running_loss / len(train_loader)
        map_bbox_50_95, map_bbox_50, map_bbox_95, map_mask_50_95, map_mask_50, map_mask_95 = evaluate(
            model, val_loader, device, epoch+1, val_images_dir
        )

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, BBox mAP@0.5: {map_bbox_50:.4f}, BBox mAP@0.95: {map_bbox_95:.4f}, \
              BBox mAP(0.5:0.95): {map_bbox_50_95:.4f}, Mask mAP@0.5: {map_mask_50:.4f}, \
              Mask mAP@0.95: {map_mask_95:.4f}, Mask mAP(0.5:0.95): {map_mask_50_95:.4f}")

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, map_bbox_50_95, map_bbox_50, map_bbox_95, 
                             map_mask_50_95, map_mask_50, map_mask_95])

        if map_bbox_50_95 > best_map:
            best_map = map_bbox_50_95
            torch.save(model.state_dict(), "best_model.pth")

    print("Training complete.")
    print(f"Best BBox mAP(0.5:0.95): {best_map:.4f}, model saved as best_model.pth")
