import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.mtcnn import fixed_image_standardization

import constants
from utils.data import listdir
from utils.facenet import Intercept
from utils.feature import Feature
from utils.user import User
from utils.map import Map

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n", required=True,
                    help="arquivo")
args = parser.parse_args()

resnet = InceptionResnetV1(pretrained='vggface2').eval()
interceptor = Intercept(resnet)

def intercept_facenet(image: np.ndarray[np.ndarray]):
    image = cv.resize(image, (160, 160), cv.INTER_AREA)
    tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
    output = interceptor.forward(tensor)

    return output

def extract_facenet_feature(image: np.ndarray[np.ndarray], layer_name: str):
    output = intercept_facenet(image)

    return output[layer_name]

def collect_dataset_features(dataset_folder_name: str):

    dataset_dir = constants.ROOT_DATASET_FOLDER / dataset_folder_name
    users = listdir(str(dataset_dir))

    map = Map()

    for idu, username in enumerate(users):
        
        user = User(name=username, dir=dataset_dir / username)

        images_paths = listdir(str(user.dir))

        map[username] = []

        for image_path in images_paths:
            image = cv.imread(image_path)
            feature_vector = extract_facenet_feature(image, 'last_bn_norm')
            feature = Feature(
                feature_vector=feature_vector, path=image_path,
                original_shape=image.shape
            )
            user.add_feature(feature)

    

if __name__ == '__main__':
    pass