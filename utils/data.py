from pathlib import Path
import pickle

def save_pickle(path, name, object):
    with open(str(path) + "/" + str(name) + ".pickle", "wb") as f:
        pickle.dump(object, f)

def listdir(dir_name: str):
    return [path for path in Path(dir_name).iterdir() if path.is_dir]

def map_paths_from_user_folder(**kwargs):
    filepaths = kwargs.get('filepaths')
    files = listdir(filepaths)
    user_maps = []
    for file in files:
        filepath = filepaths + file
        user_maps.append(filepath)

    return user_maps

def map_paths_from_folder(**kwargs):
    directory = kwargs.get('directory')
    names = listdir(directory)
    maps = {}
    for name in names:
        filepaths = directory + "/" + name + "/"
        user_maps = map_paths_from_user_folder(filepaths=filepaths)
        maps.update({name: user_maps})

    return maps