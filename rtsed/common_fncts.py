import sys
import pandas as pd


def read_input_file(input_file):
    try:
        input_data = pd.read_csv(input_file)
    except:
        print("Problem reading {}. Please check if the file exists and it is in the required format.")
        sys.exit()

    return input_data


def merge_dicts(dicts):
    """Given list of dicts, merge them into a new dict as a shallow copy.
    :param dicts: list of dictionaries
    :return: merged dictionary
    """
    out_dict = {}
    for current_dict in dicts:
        if not isinstance(current_dict, dict):
            continue
        out_dict = out_dict.copy()
        try:
            out_dict.update(current_dict)
        except:
            print('error append dict')
    return out_dict


def diff_keys(dict1, dict2):
    """
    return dictionary 1 without keys from dictionary 2
    :param dict1:
    :param dict2:
    :return:
    """
    keys = dict2.keys()
    return without_keys(dict1, keys)
