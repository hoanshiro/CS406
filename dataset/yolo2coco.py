import json
from pylabel import importer


def convert2coco(img):
    path_to_annotations = "/home/henry/Project/DIP/dataset/test_set/labels"
    # Identify the path to get from the annotations to the images
    path_to_images = "/home/henry/Project/DIP/dataset/test_set/images"
    # Import the dataset into the pylable schema
    # Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
    traffic_sign_classes = ['No entry', 'No parking / waiting', 'No turning', 'Max Speed',
                            'Other prohibition signs', 'Warning', 'Mandatory']
    dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images,
                                    cat_names=traffic_sign_classes, img_ext="png", name="coco128")

    dataset.export.ExportToCoco(cat_id_index=1,
                                output_path="/home/henry/Project/DIP/dataset/test_set/test.json")


def convert2valid_format(label_path):
    with open(label_path, "r") as f:
        json_data = json.load(f)
    valid_label = []
    for item in json_data["images"]:
        item["annotations"] = []
        for annotation in json_data["annotations"]:
            if item["id"] == annotation["image_id"]:
                item["annotations"].append(annotation)
        valid_label.append(item)
    with open("/home/henry/Project/DIP/dataset/test_set/test_valid.json", "w") as f:
        json.dump(valid_label, f)


if __name__ == "__main__":
    convert2valid_format("/home/henry/Project/DIP/dataset/test_set/test.json")