import numpy as np
import cv2
from tqdm import tqdm
import random
import json

path = "/DATA2/chenxi/MinkowskiEngine/data/gpd_data_v4/"
# path = ""

all_img = []

file_name = path + "test_seen/labels_test_seen.npy"
labels = np.load(file_name)

positive_img = []
negative_img = []
img_cnt = 0

for label in labels:
    if 0 < label < 0.11:
        positive_img.append(path + "test_seen/image/" + str(img_cnt).zfill(6) + ".jpg")
    else:
        negative_img.append(path + "test_seen/image/" + str(img_cnt).zfill(6) + ".jpg")
    img_cnt += 1

positive_cnt = len(positive_img)
negative_cnt = len(negative_img)
print("# positive images:", positive_cnt, "# negative images:", negative_cnt)

if positive_cnt < negative_cnt:
    final_size = positive_cnt
    for i in range(final_size):
        all_img.append([positive_img[i], 1])
        all_img.append([negative_img[i], 0])
else:
    final_size = negative_cnt
    for i in range(final_size):
        all_img.append([positive_img[i], 1])
        all_img.append([negative_img[i], 0])

print('# images chosen:', len(all_img))

random.shuffle(all_img)

img_path_list = []
label_list = []

for image_data in tqdm(all_img):
    image_path, image_label = image_data[0], image_data[1]
    img_path_list.append(image_path)
    label_list.append(image_label)

f = {'images': img_path_list, 'labels': label_list}

with open("../mydata/test.json", "w") as file:
    json.dump(f, file)



