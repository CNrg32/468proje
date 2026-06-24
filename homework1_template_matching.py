import os
import json
import random
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc

DATA_DIR = Path("data/raw")
IMAGE_DIR = DATA_DIR / "images"
ANNOTATION_FILE = DATA_DIR / "annotations.json"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
TARGET_CLASS_NAME= "halo"
IMAGE_SIZE =256
RANDOM_SEED = 42
TEMPLATE_SCALES={128,64,32}
MATCH_THRESHOLDS={0.2,0.3,0.4,0.5,0.6,0.7}
AREA_THRESHOLDS={0.1,0.2,0.3,0.4,0.5}
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_coco_annotations(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        images = {image["id"]: image for image in data["images"]}
        categories = {category["id"]: category["name"] for category in data["categories"]}
        annotations_by_image = {}
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            annotations_by_image.setdefault(image_id, []).append(annotation)
        print("Available categories:")
        for category_id, category_name in categories.items():
            print(f"  {category_id}: {category_name}")
        return images, categories, annotations_by_image

def find_category_id(categories, target_class_name):
    for category_id, category_name in categories.items():
        if category_name == target_class_name:
            return category_id
    raise ValueError(f"Category '{target_class_name}' not found in categories.")

def read_gray_resized(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    resized=cv2.resize(gray,(IMAGE_SIZE,IMAGE_SIZE))
    resized=cv2.equalizeHist(resized)
    edges=cv2.Canny(resized,50,100)
    return edges

def scale_bbox_to_256(bbox, original_width, original_height):
    x, y, width, height = bbox
    sx = IMAGE_SIZE / original_width
    sy = IMAGE_SIZE / original_height

    x1 = int(x * sx)
    y1 = int(y * sy)
    x2 = int((x + width) * sx)
    y2 = int((y + height) * sy)
    x1=max(0, min(x1, IMAGE_SIZE - 1))
    y1=max(0, min(y1, IMAGE_SIZE - 1))
    x2=max(0, min(x2, IMAGE_SIZE - 1))
    y2=max(0, min(y2, IMAGE_SIZE - 1))
    return (x1, y1, x2, y2)

def xcrop_object_from_image(gray_image, bbox):
    x1, y1, x2, y2 = bbox
    crop=gray_image[y1:y2, x1:x2]

    if crop.size == 0:
        return None
    crop=cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
    return crop

def get_records(images,annotations_by_image,target_category_id):
    records=[]
    for image_id, image_info in images.items():
        file_name=image_info["file_name"]
        image_path=IMAGE_DIR / file_name
        if not image_path.exists():
            continue
        width=image_info["width"]
        height=image_info["height"]
        anns=annotations_by_image.get(image_id, [])
        target_boxes=[]
        for ann in anns:
            if ann["category_id"]==target_category_id:
                bbox=scale_bbox_to_256(ann["bbox"], width, height)
                target_boxes.append(bbox)
        records.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_path": image_path,
            "target_boxes": target_boxes,
            "has_object": len(target_boxes) > 0
        })
    return records

def split_dataset(records):
    positive=[r for r in records if r["has_object"]]
    negative=[r for r in records if not r["has_object"]]
    random.shuffle(positive)
    random.shuffle(negative)
    train_positive=positive[:10]
    validation_positive=positive[10:210]
    validation_negative=negative[:600]
    test_positive=positive[210:410]
    test_negative=negative[600:1200]
    validation=validation_positive + validation_negative
    test=test_positive + test_negative
    random.shuffle(validation)
    random.shuffle(test)

    print(f"Train positive templates: {len(train_positive)}")
    print(f"Validation positive templates: {len(validation_positive)}")
    print(f"Validation negative templates: {len(validation_negative)}")
    print(f"Test positive templates: {len(test_positive)}")
    print(f"Test negative templates: {len(test_negative)}")

    if len(train_positive) < 10:
        raise ValueError("Not enough positive samples for training. Please ensure there are at least 10 positive samples.")
    if len(validation_positive) < 200 or len(validation_negative) < 600:
        raise ValueError("Not enough samples for validation. Please ensure there are at least 200 positive and 600 negative samples.")
    if len(test_positive) < 200 or len(test_negative) < 600:
        raise ValueError("Not enough samples for testing. Please ensure there are at least 200 positive and 600 negative samples.")
    return train_positive, validation, test

def build_templates(train_records):
    templates=[]
    for record in train_records:
        gray=read_gray_resized(record["image_path"])
        if gray is None:
            continue
        if not record["target_boxes"]:
            continue
        crop=xcrop_object_from_image(gray, record["target_boxes"][0])
        if crop is not None:
            templates.append(crop)
    return templates

def calculate_overlap_percentage(pred_box,true_boxes):
    if pred_box is None or len(true_boxes) == 0:
        return 0.0
    px1,py1,px2,py2=pred_box
    pred_area=max(0,px2-px1)*max(0,py2-py1)
    if pred_area == 0:
        return 0.0
    best_percentage=0.0
    for box in true_boxes:
        tx1,ty1,tx2,ty2=box
        inter_x1=max(px1,tx1)
        inter_y1=max(py1,ty1)
        inter_x2=min(px2,tx2)
        inter_y2=min(py2,ty2)
        intersection=max(0,inter_x2-inter_x1)*max(0,inter_y2-inter_y1)
        true_area=max(0,tx2-tx1)*max(0,ty2-ty1)
        if true_area >0:
            percentage=intersection/true_area
            best_percentage=max(best_percentage,percentage)
    return best_percentage

def template_match_single_image(gray_image,templates,match_threshold):
    best_score=-1.0
    best_box=None
    for template in templates:
        for scale in TEMPLATE_SCALES:
            if scale>IMAGE_SIZE:
                continue
            scaled_template=cv2.resize(template,(scale,scale))
            if scaled_template.shape[0]>gray_image.shape[0] or scaled_template.shape[1]>gray_image.shape[1]:
                continue
            result=cv2.matchTemplate(gray_image,scaled_template,cv2.TM_CCOEFF_NORMED)
            _,max_val,_,max_loc=cv2.minMaxLoc(result)
            if max_val>best_score:
                x1,y1=max_loc
                x2=x1+scale
                y2=y1+scale
                best_score=max_val
                best_box=(x1,y1,x2,y2)
    if best_score>=match_threshold:
        return True,best_box,best_score
    return False,None,best_score

def evaluate(records,templates,match_threshold,area_threshold):
    y_true=[]
    y_pred=[]
    details=[]
    for record in records:
        gray=read_gray_resized(record["image_path"])
        if gray is None:
            continue
        predicted_object,pred_box,score =template_match_single_image(gray,templates,match_threshold)
        true_object=record["has_object"]
        if predicted_object:
            overlap_percentage=calculate_overlap_percentage(pred_box,record["target_boxes"])
            if record["has_object"]:
                final_prediction = predicted_object and overlap_percentage >= area_threshold
            else:
                final_prediction = predicted_object
        else:
            final_prediction=False
            overlap_percentage=0.0
        
        y_true.append(1 if true_object else 0)
        y_pred.append(1 if final_prediction else 0)
        details.append({
            "file_name": record["file_name"],
            "true_object": true_object,
            "predicted": final_prediction,
            "score": score,
            "overlap_percentage": overlap_percentage,
            "box":pred_box
        })
    metrics={
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,zero_division=0),
        "recall": recall_score(y_true,y_pred,zero_division=0),
        "f1_score": f1_score(y_true,y_pred,zero_division=0),
        "confusion_matrix": confusion_matrix(y_true,y_pred).tolist(),
        "classification_report": classification_report(y_true,y_pred,zero_division=0,output_dict=True)
    }
    return metrics,details

def precompute_matches(records, templates):
    cache = []
    for record in records:
        gray = read_gray_resized(record["image_path"])
        if gray is None:
            continue
        _, box, score = template_match_single_image(gray, templates, -1.0)
        overlap = calculate_overlap_percentage(box, record["target_boxes"])
        cache.append((record, score, overlap))
    return cache

def evaluate_cached(cache, match_threshold, area_threshold):
    y_true, y_pred = [], []
    for record, score, overlap in cache:
        predicted_object = score >= match_threshold
        if record["has_object"]:
            final_prediction = predicted_object and overlap >= area_threshold
        else:
            final_prediction = predicted_object
        y_true.append(1 if record["has_object"] else 0)
        y_pred.append(1 if final_prediction else 0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    }

def optimize_thresholds(validation_records,templates):
    best_results=None
    all_rows=[]
    cache = precompute_matches(validation_records, templates)
    for match_threshold in MATCH_THRESHOLDS:
        for area_threshold in AREA_THRESHOLDS:
            metrics=evaluate_cached(cache, match_threshold, area_threshold)
            all_rows.append({"match":match_threshold, "area": area_threshold, "accuracy": metrics["accuracy"],
                              "precision": metrics["precision"], "recall": metrics["recall"],
                                "f1_score": metrics["f1_score"]})
            current={"match_threshold": match_threshold, "area_threshold": area_threshold, "metrics": metrics
                     }
            if best_results is None or metrics["f1_score"]>best_results["metrics"]["f1_score"]:
                best_results=current
            print(
                f"match={match_threshold}, area={area_threshold}, accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1_score={metrics['f1_score']:.4f}"
                
            )
            save_metrics(f"validation_match_{match_threshold}_area_{area_threshold}", current)
    save_metrics("validation_grid", all_rows)
    return best_results

            
def save_metrics(name,result):
    output_path=RESULTS_DIR / f"{name}_metrics.json"
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(f"Saved metrics to {output_path}")

def save_detection_visual(record, templates, match_threshold):
    image = cv2.imread(str(record["image_path"]))
    edges = read_gray_resized(record["image_path"])      # eşleştirme girişi = kenar
    predicted_object, pred_box, score = template_match_single_image(
        edges, templates, match_threshold
    )
    resized_color = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))   # çizim için renkli
    if predicted_object and pred_box is not None:
        x1, y1, x2, y2 = pred_box
        cv2.rectangle(resized_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized_color, f"{TARGET_CLASS_NAME} ({score:.2f})",
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(RESULTS_DIR / "sample_detection.jpg"), resized_color)
        print(f"Saved sample detection visualization to {RESULTS_DIR / 'sample_detection.jpg'}")

def main():
    images, categories, annotations_by_image = load_coco_annotations(ANNOTATION_FILE)
    target_category_id = find_category_id(categories, TARGET_CLASS_NAME)

    records = get_records(images, annotations_by_image, target_category_id)
    train_records, validation_records, test_records = split_dataset(records)

    templates = build_templates(train_records)
    print(f"Template count: {len(templates)}")

    print("\nValidation threshold search:")
    best_validation = optimize_thresholds(validation_records, templates)
    save_metrics("validation_best", best_validation)

    best_match_threshold = best_validation["match_threshold"]
    best_area_threshold = best_validation["area_threshold"]

    print("\nTesting with best thresholds:")
    test_metrics, test_details = evaluate(
        test_records,
        templates,
        best_match_threshold,
        best_area_threshold
    )

    test_result = {
        "match_threshold": best_match_threshold,
        "area_threshold": best_area_threshold,
        "metrics": test_metrics,
        "details": test_details
    }

    print(test_metrics["classification_report"])
    print("Confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    save_metrics("test", test_result)

    positive_test_records = [r for r in test_records if r["has_object"]]
    if positive_test_records:
        save_detection_visual(positive_test_records[0], templates, best_match_threshold)


if __name__ == "__main__":
    main()