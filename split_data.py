from utils.data import map_paths_from_folder
import os, shutil, argparse
import numpy as np
from copy import deepcopy as dc
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n", required=True,
                    help="arquivo")
args = parser.parse_args()
PERCENTUAL = 0.75
TRAIN = '/home/yves/Imagens/data/train-{}/'.format(args.n)
TEST = '/home/yves/Imagens/data/test-{}/'.format(args.n)
os.mkdir(TRAIN)
os.mkdir(TEST)

database = map_paths_from_folder(directory='/home/yves/Área de Trabalho/database')

for name, files in database.items():
    os.mkdir(TRAIN + name)
    os.mkdir(TEST + name)
    TAM = len(files)
    vectorFiles = dc(files)
    np.random.shuffle(vectorFiles)
    trainFiles = vectorFiles[:int(TAM * PERCENTUAL)]
    testFiles = vectorFiles[int(TAM * PERCENTUAL):]
    for tf in trainFiles:
        title = tf.split("/")[-1]
        shutil.copyfile(tf, TRAIN + name + "/" + title)

    for tf in testFiles:
        title = tf.split("/")[-1]
        shutil.copyfile(tf, TEST + name + "/" + title)