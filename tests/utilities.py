import inspect
import os


def get_cached_revisions():
    test_dir = os.path.dirname(
                    os.path.abspath(inspect.getfile(inspect.currentframe()))
                    )
    data_dir = 'cached_revs'
    data_dir_path = os.path.join(test_dir, data_dir)
    rev_texts = {}
    for filename in os.listdir(data_dir_path):
        filepath = os.path.join(data_dir_path, filename)
        with open(filepath) as f:
            rev_texts[int(filename)] = f.read()
    return rev_texts
