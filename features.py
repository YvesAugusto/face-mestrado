import argparse

import cv2 as cv
import numpy as np

import constants
from utils.data import listdir, save_pickle
from utils.facenet import load_facenet
from utils.feature import Feature
from utils.map import Map, TestMap
from utils.user import User

facenet_model = load_facenet()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", required=True,
                    help="nome pasta com arquivos da base de dados(será acessado a partir da referência ROOT_PATH setada no arquivo de constantes)")
args = parser.parse_args()

def extract_facenet_feature(image: np.ndarray[np.ndarray]):
    rsz = cv.resize(image, (150, 150), cv.INTER_AREA)
    rsz = np.reshape(rsz, (1, rsz.shape[0], rsz.shape[1], rsz.shape[2]))
    rsz = (rsz - 127.5)/128

    return facenet_model.predict(rsz)

def collect_dataset_features(dataset_folder_name: str):

    dataset_dir = constants.ROOT_DATASET_FOLDER / dataset_folder_name
    users = listdir(str(dataset_dir))

    test = dataset_folder_name.split("-")[0] == "test"
    map = Map(dataset_dir)
    if test:
        map = TestMap(dataset_dir)
    
    p = 0
    for idu, username in enumerate(users):
        p+=1
        user = User(name=str(username).split("/")[-1], dir=dataset_dir / username)

        images_paths = listdir(str(user.dir))
        for image_path in images_paths:

            image = cv.imread(str(image_path))
            feature_vector = extract_facenet_feature(image)
            feature = Feature(
                feature_vector=feature_vector, path=image_path,
                original_shape=image.shape
            )
            user.add_feature(feature)
        print("Extracted users: {}/{}".format(p, len(users)))
        map.add_user(user)
    
    save_pickle(constants.FEATURES_PATH, dataset_folder_name, map)

    return map
    

if __name__ == '__main__':

    collect_dataset_features(args.folder)