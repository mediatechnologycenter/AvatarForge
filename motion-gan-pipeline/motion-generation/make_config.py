import yaml
import os


def make_config(name, dictionary):
    config_path = './config'
    yml_path = os.path.join(config_path, f'{name}.yml')
    with open(yml_path, 'w') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False)


def base_config():

    config = {}

    ## Pick a name
    # name = 'MTC'
    # name = 'TED384-v2'
    name = 'TMP_AVSpeech'
    
    ## Pick a task
    task = 'Audio2Headpose'
    # task = 'Correlation'

    ## Pick a dataset
    dataset = 'deepspeech'
    # dataset = 'emotion'

    # Parameters to setup experiment.
    config['experiment'] = {
        'name': f'{task}_{name}',
        'dataset_mode': dataset,
        'dataroot': '/media/apennino/',
        'dataset_names': name,
        'fps': 25,
    }

    make_config(config['experiment']['name'], config)


if __name__ == "__main__":
    base_config()
