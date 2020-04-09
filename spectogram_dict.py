from spectrograms import stft_matrix
import os
from dev_data import df_dev




def build_class_dictionaries(dir_name):
    depressed_dict = dict()
    normal_dict = dict()
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                if in_dev_split(partic_id):
                    wav_file = os.path.join(subdir, file)
                    
                    mat = stft_matrix(wav_file)
                    depressed = get_depression_label(partic_id)  # 1 if True
                    if depressed:
                        depressed_dict[partic_id] = mat
                    elif not depressed:
                        normal_dict[partic_id] = mat
    return depressed_dict, normal_dict


def in_dev_split(partic_id):
    return partic_id in set(df_dev['Participant_ID'].values)


def get_depression_label(partic_id):
    
    return df_dev.loc[df_dev['Participant_ID'] ==
                      partic_id]['PHQ8_Binary'].item()


if __name__ == '__main__':
    dir_name = '../../data/interim'
    depressed_dict, normal_dict = build_class_dictionaries(dir_name)
