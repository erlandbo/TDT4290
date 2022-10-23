"""
Yaml configuration generation script
Reads a default yaml file and create multiple
new ones with specified alterations to variables.
Is done to aid in grid-searching for hyper-param
and configuration tuning.
#TODO: Make this script easier to use (if possible)
"""
import yaml
import copy
import numpy as np


def yaml_to_dict(yaml_path):
    yaml_file = open(yaml_path, 'r')
    text = yaml_file.read()
    yaml_file.close()
    yaml_as_dict = yaml.load(text)
    return yaml_as_dict



def write_value(my_dict, nesting_sequence, value):
    #Quick function to write recursive dict
    if len(nesting_sequence) == 1:
        my_dict[nesting_sequence[0]] = value
        return my_dict
    else:
        my_dict[nesting_sequence[0]] = write_value(
            my_dict[nesting_sequence[0]],
            nesting_sequence[1:],
            value
        )
        return my_dict


def multi_dict_creator(
    nesting_sequence : list,
    values : list,
    default_dict : dict
    ):
    """
    Function to create bunch of dictionaries with one variable altered
    Arguments :
    nesting sequence - sequence of key-dict nesting to get to value
    values - Values to write to nesting sequence value
    default_dict - default dictionary

    There is probably a better way to do this, but this was quick & easy
    """
    out_dicts = []
    for value in values:
        new_dict = default_dict
        new_dict = write_value(new_dict, nesting_sequence, value)
        out_dict = copy.deepcopy(new_dict)
        out_dicts.append(out_dict)
    return out_dicts


def dict_to_yaml(out_path, out_dict):
    with open(out_path, 'w') as outfile:
        yaml.dump(out_dict, outfile, default_flow_style=False)

def generate_configs(
    nesting_sequence:list,
    default_yaml_path:str,
    values:list,
    value_name : str,
    out_path : str
    ):
    """
    Function to generate multiple yaml files with minor alterations
    Arguments:
    nesting sequence - list of keys do get to altered value
    values - values to overwrite with
    value_name - name of altered value to be used for output filenames
    """
    default_dict = yaml_to_dict(yaml_path=default_yaml_path)
    new_dicts = multi_dict_creator(nesting_sequence, values, default_dict)
    for value, new_dict in zip(values, new_dicts):
        out_name = default_yaml_path.split(".")[0] + \
        "_" + value_name + str(value) + ".yaml"
        print(new_dict)
        dict_to_yaml(out_path + "/" + out_name, new_dict)


        

def main():
    #Example usage
    nesting_list = ["INPUT", "TRANSFORM", "RAND_GAUSS", "INTENSITY"]
    values = list(np.linspace(0.05, 2, 40))
    yaml_path = "configs/resnet50.yaml"
    out_path = "configs/rand_gauss"
    name = "rg"
    generate_configs(
        nesting_sequence=nesting_list,
        default_yaml_path=yaml_path,
        values=values,
        value_name=name,
        out_path=out_path
    )
        

if __name__ == "__main__":
    main()
