import argparse
import os
import yaml
import numpy as np
import pandas as pd
from utils.path_helper import set_project_root_path

if __name__ == '__main__':
    set_project_root_path()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration_file',
        '-cf',
        help='The path of the configuration file',
        default="configs/test_set_voting.yaml")
    args = parser.parse_args()
    configuration_file = str(args.configuration_file)
    with open(configuration_file, 'r') as configs_file:
        configs = yaml.safe_load(configs_file)
    prediction_files = os.listdir(configs['predictions_path'])
    threshold = configs['threshold']
    output_path = configs['voting_output_path']
    headers = ["Argument ID"] + configs["labels_column"]

    voting_array = np.zeros((configs['set_size'], 20), dtype=float)
    argument_ids = None
    for prediction_file in prediction_files:
        if prediction_file.endswith('tsv'):
            prediction_file = os.path.join(configs['predictions_path'], prediction_file)
            predictions = pd.read_csv(filepath_or_buffer=prediction_file, sep='\t')
            if argument_ids is None:
                argument_ids = predictions["Argument ID"].to_numpy()
            predictions = predictions.iloc[:, 1:].to_numpy()
            voting_array += predictions
    voting_array /= (len(prediction_files) - 1)
    voting_array = np.where(voting_array > threshold, 1, 0)
    final_voting = np.asarray(voting_array, dtype=str)
    final_voting = np.column_stack((argument_ids, final_voting))
    np.savetxt(
        fname=output_path,
        X=final_voting,
        fmt='%s',  # save as int (defaults to float)
        delimiter='\t',
        header='\t'.join(headers),
        comments=''
    )
