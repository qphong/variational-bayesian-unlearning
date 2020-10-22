import pickle 
import os.path


def load_file_if_exist_else_create(filename, create_file_func, force_create_new=False):

    if os.path.isfile(filename) and not force_create_new:
        print("Load from {}".format(filename))
        file_content = pickle.load(open(filename, 'rb'))

    else:
        if force_create_new:
            print("Create {}".format(filename))
        else:
            print("{} does not exists. Creating file.".format(filename))

        file_content = create_file_func()
        pickle.dump(file_content, open(filename, 'wb'))

    return file_content
