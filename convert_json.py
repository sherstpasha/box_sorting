import os
import json
import random
import numpy as np
from sklearn.utils import shuffle
import cv2

def extract_coordinates(box):
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    return [x, y, x + w, y, x + w, y + h, x, y + h]

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    boxes = data.get('boxes', [])
    coordinates = [extract_coordinates(box) for box in boxes]

    return coordinates

def get_max_dimensions(coordinates):
    max_x = max(max(coord[::2]) for coord in coordinates)
    max_y = max(max(coord[1::2]) for coord in coordinates)
    img_width = int(max_x * 2)
    img_height = int(max_y * 2)
    return img_width, img_height

def rotate_point(x, y, angle, cx, cy):
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    x -= cx
    y -= cy

    x_new = x * cos_theta + y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    x_new += cx
    y_new += cy

    return x_new, y_new

def rotate_boxes(coordinates, angle):
    rotated_coords = []
    cx = np.mean([coord[0] for coord in coordinates])  # Центр вращения по x
    cy = np.mean([coord[1] for coord in coordinates])  # Центр вращения по y

    for coord in coordinates:
        rotated_coord = []
        for i in range(0, len(coord), 2):
            x_new, y_new = rotate_point(coord[i], coord[i+1], angle, cx, cy)
            rotated_coord.extend([x_new, y_new])
        rotated_coords.append(rotated_coord)
    return rotated_coords

def transform_boxes(coordinates, top_left, top_right, bottom_left, bottom_right, img_width, img_height):
    w, h = img_width, img_height
    max_hor_shift = w // 2
    max_ver_shift = h // 2

    tl_shift = top_left * max_hor_shift
    tr_shift = top_right * max_hor_shift
    bl_shift = bottom_left * max_ver_shift
    br_shift = bottom_right * max_ver_shift

    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32([
        [tl_shift, tl_shift],
        [w - tr_shift, tr_shift],
        [w - br_shift, h - br_shift],
        [bl_shift, h - bl_shift]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_coords = []

    for coord in coordinates:
        points = np.array([[coord[0], coord[1]], [coord[2], coord[3]], [coord[4], coord[5]], [coord[6], coord[7]]], dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(np.array([points]), matrix)[0]
        transformed_coord = transformed_points.flatten().tolist()
        transformed_coords.append(transformed_coord)

    return transformed_coords

def augment_data(coordinates, img_width, img_height):
    augmented_data = []

    for _ in range(50):
        angle = random.uniform(-7, 7)
        top_left = random.uniform(-0.1, 0.1)
        top_right = random.uniform(-0.1, 0.1)
        bottom_left = random.uniform(-0.1, 0.1)
        bottom_right = random.uniform(-0.1, 0.1)

        # Применение вращения
        rotated_coords = rotate_boxes(coordinates, angle)

        # Применение искажений
        transformed_coords = transform_boxes(rotated_coords, top_left, top_right, bottom_left, bottom_right, img_width, img_height)

        # Перемешивание
        shuffled_coords, indices = shuffle(transformed_coords, np.arange(len(transformed_coords)), random_state=0)
        
        augmented_data.append((shuffled_coords, indices))

    return augmented_data

def write_to_file(coordinates, indices, output_coord_path, output_index_path):
    with open(output_coord_path, 'w', encoding='utf-8') as coord_file, open(output_index_path, 'w', encoding='utf-8') as index_file:
        for coord, index in zip(coordinates, indices):
            coord_file.write(" ".join(map(str, coord)) + "\n")
            index_file.write(f"{index}\n")

def main(input_folder, output_base_folder):
    # Создание папок для данных
    split_folders = ['train', 'val', 'test']
    for split in split_folders:
        coord_folder = os.path.join(output_base_folder, split, 'coordinates')
        index_folder = os.path.join(output_base_folder, split, 'indices')
        os.makedirs(coord_folder, exist_ok=True)
        os.makedirs(index_folder, exist_ok=True)
    
    # Получение списка всех файлов и их перемешивание
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    random.shuffle(all_files)

    # Определение разделения
    num_files = len(all_files)
    train_split = int(0.7 * num_files)
    val_split = int(0.15 * num_files)
    test_split = num_files - train_split - val_split

    splits = {
        'train': all_files[:train_split],
        'val': all_files[train_split:train_split + val_split],
        'test': all_files[train_split + val_split:]
    }

    for split, files in splits.items():
        for filename in files:
            file_path = os.path.join(input_folder, filename)
            coordinates = process_json_file(file_path)
            
            img_width, img_height = get_max_dimensions(coordinates)
            
            augmented_data = augment_data(coordinates, img_width, img_height)
            
            base_name = os.path.splitext(filename)[0]
            coord_folder = os.path.join(output_base_folder, split, 'coordinates')
            index_folder = os.path.join(output_base_folder, split, 'indices')
            for i, (coords, indices) in enumerate(augmented_data):
                output_coord_path = os.path.join(coord_folder, f"{base_name}_coordinates_{i}.txt")
                output_index_path = os.path.join(index_folder, f"{base_name}_indices_{i}.txt")
                
                write_to_file(coords, indices, output_coord_path, output_index_path)
                print(f"Processed {filename} (augmentation {i}) and saved to {output_coord_path} and {output_index_path}")

if __name__ == "__main__":
    input_folder = r"C:\Users\user\Desktop\examples\boxes"  # Укажите путь к папке с JSON файлами
    output_base_folder = r"C:\Users\user\Desktop\examples\data"  # Укажите путь к базовой папке для сохранения выходных файлов
    main(input_folder, output_base_folder)
