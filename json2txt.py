import json
import os
import cv2
import glob


def convert_labelme_to_yolo_pose(json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    keypoint_names = ["head", "body1", "body2", "body3", "tail"]

    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_width = data.get('imageWidth')
        image_height = data.get('imageHeight')

        if image_width is None or image_height is None:
            img_filename = data.get('imagePath', os.path.basename(json_path).replace('.json', '.png'))
            img_path = os.path.join(json_dir, img_filename)

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    image_height, image_width = img.shape[:2]

            if image_width is None:
                print(f"警告: 找不到 {json_path} 对应的图片尺寸，已跳过。")
                continue

        boxes = []
        points = []

        # 提取框和关键点
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle' and shape['label'] == 'worm':
                boxes.append(shape['points'])
            elif shape['shape_type'] == 'point':
                points.append({
                    'label': shape['label'],
                    'coord': shape['points'][0]
                })

        lines = []
        for box in boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)

            xc = (xmin + xmax) / 2.0 / image_width
            yc = (ymin + ymax) / 2.0 / image_height
            w = (xmax - xmin) / image_width
            h = (ymax - ymin) / image_height

            kpts = {name: None for name in keypoint_names}
            for pt in points:
                px, py = pt['coord']
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    kpts[pt['label']] = pt['coord']

            line = f"0 {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}"
            for name in keypoint_names:
                coord = kpts[name]
                if coord is not None:
                    kx = coord[0] / image_width
                    ky = coord[1] / image_height
                    line += f" {kx:.5f} {ky:.5f} 2"
                else:
                    line += " 0.00000 0.00000 0"

            lines.append(line)

        txt_filename = os.path.basename(json_path).replace('.json', '.txt')
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"成功转换: {json_path} -> {txt_path}")


if __name__ == '__main__':
    # Your dataset
    json_directory = './'
    output_directory = './'
    convert_labelme_to_yolo_pose(json_directory, output_directory)