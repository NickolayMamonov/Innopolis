import cv2
import xml.etree.ElementTree as ET
import glob
import os
import json

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

classes = []
input_dir = "/Users/fegol/Documents/Innopolis/annotations_loaded"


files = glob.glob(os.path.join(input_dir, '*.xml'))
for fil in files:
    basename = os.path.basename(fil) #road0.xml
    filename = os.path.splitext(basename)[0] #road0

    result = []
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text
        if label not in classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")

    if result:
        from random import randint
        number = randint(1, 100)
        if number <= 2:
            with open(os.path.join("/Users/fegol/Documents/Innopolis/labels/val/", f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))
            os.replace(f"/Users/fegol/Documents/Innopolis/images_loaded/{filename}.png",
                       f"/Users/fegol/Documents/Innopolis/images/val/{filename}.png")
        elif number <= 22:
            with open(os.path.join("/Users/fegol/Documents/Innopolis/labels/test/",f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))
            os.replace(f"/Users/fegol/Documents/Innopolis/images_loaded/{filename}.png",
                       f"/Users/fegol/Documents/Innopolis/images/test/{filename}.png")
        else:
            with open(os.path.join("/Users/fegol/Documents/Innopolis/labels/train/", f"{filename}.txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(result))
            os.replace(f"/Users/fegol/Documents/Innopolis/images_loaded/{filename}.png",
                       f"/Users/fegol/Documents/Innopolis/images/train/{filename}.png")

with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))