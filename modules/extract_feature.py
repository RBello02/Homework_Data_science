# it takes the feature from the waw file and join it whit the cvc file that we have

# come useful libraries for deature extractionP
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import librosa
import os



# function which reads a audio file and extract some useful spectral features 
def audio_feature_extraction(dataset, num_segmentation):
    
    mfcc_list = []
    num_coeff_mfcc = 10

    # For each audio file we are going to extract the MFCC coefficients and some stats of the spectogram
    for sr, file_path in zip(dataset['sampling_rate'], dataset['path']):
        path = 'datasets/' + file_path
        print(path)
        
        # Reading the audio file 
        audio_time_series, sampling_rate = librosa.load(path = path, sr = sr)
        spectrogram = librosa.feature.melspectrogram(y= audio_time_series, sr= sampling_rate, n_mels=40)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # deviding the spettrogram in num_segmentation parts e take the mean of each part and the variance
        n = num_segmentation
        m = log_spectrogram.shape[1] // n
        

        for i in range(n):
            part = log_spectrogram[:, i*m:(i+1)*m]
            mean = np.mean(part)
            variance = np.var(part)
            dataset.loc[dataset['path'] == file_path, f'mean_spectrogram_{i+1}'] = mean
            dataset.loc[dataset['path'] == file_path, f'var_spectrogram_{i+1}'] = variance
            
        
        # Extracting MFCC coefficients
        mfcc = librosa.feature.mfcc(y= audio_time_series , sr= sampling_rate, n_mfcc= num_coeff_mfcc)  
        mfcc_list.append(np.mean(mfcc, axis = 1))
        
        
    # Converting the mfcc list in a dataframe to concatenate it to dataset
    mfcc_df = pd.DataFrame(mfcc_list, columns=[f'mfcc_{i+1}' for i in range(num_coeff_mfcc)])
    new_dataset = pd.concat([dataset, mfcc_df], axis=1)

    return new_dataset


def new_csv_with_audio_extracted_features(num_segmentation = 20):
    
    ## LOADING THE DEVELOPMENT DATASET
    os.chdir('..')
    filename_dev = 'datasets/development.csv'     
    
    dir = os.getcwd()
    print('Current dir = ', dir)
    print('UFFAAAA')
    
    # Loading the dataset using Pandas dataframe
    data_dev = pd.read_csv(filename_dev)


    # First, let's convert tempo in a float data type
    data_dev['tempo'] = data_dev['tempo'].str.replace('[','')
    data_dev['tempo'] = data_dev['tempo'].str.replace(']','').astype(float)

    # One-hot encoding for attribute gender and ethnicity
    # Defyning the encoder for the attribute 'ethnicity'
    all_categories_etn = sorted(set(data_dev['ethnicity']))
    encoder_etn = OneHotEncoder(categories=[all_categories_etn], handle_unknown='ignore')
    etn_encoded = encoder_etn.fit_transform(data_dev[['ethnicity']]).toarray()
    etn_encoded_df = pd.DataFrame(etn_encoded, columns=['ethnicity_' + cat for cat in all_categories_etn])

    # Defyning the encoder for the attribute 'gender'
    all_categories_gender = sorted(set(data_dev['gender']))
    encoder_gender = OneHotEncoder(categories=[all_categories_gender], handle_unknown='ignore')
    gender_encoded = encoder_gender.fit_transform(data_dev[['gender']]).toarray()
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=['gender_' + cat for cat in all_categories_gender])

    # Putting back toghether the dataset
    data_dev = pd.concat([data_dev.reset_index(drop=True).drop(columns=['ethnicity']), etn_encoded_df], axis=1)
    data_dev = pd.concat([data_dev.reset_index(drop=True).drop(columns=['gender']), gender_encoded_df], axis=1)


    # Extracting the audio features
    data_dev = audio_feature_extraction(data_dev, num_segmentation)

    # return a cvs file with the new features
    data_dev.to_csv(f'datasets/development_features_{num_segmentation}.csv', index = False)
    print(f'File saved as development_features_{num_segmentation}_prova.csv')
    
    
    
    
    ## LOADING THE EVALUATION DATSET
    filename = 'datasets/evaluation.csv'     
    
    # Loading the dataset using Pandas dataframe
    data_eval = pd.read_csv(filename)

    data_eval['tempo'] = data_eval['tempo'].str.replace('[','')
    data_eval['tempo'] = data_eval['tempo'].str.replace(']','').astype(float)

    # encoding gender and etnhicity with the encoder preoviously created
    # Encoding per "ethnicity"
    etn_encoded = encoder_etn.transform(data_eval[['ethnicity']]).toarray()
    etn_encoded_df = pd.DataFrame(etn_encoded, columns=['ethnicity_' + cat for cat in all_categories_etn])

    # Encoding per "gender"
    gender_encoded = encoder_gender.transform(data_eval[['gender']]).toarray()
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=['gender_' + cat for cat in all_categories_gender])

    # Putting back togheter the dataste
    data_eval = pd.concat([data_eval.reset_index(drop=True).drop(columns=['ethnicity']), etn_encoded_df], axis=1)
    data_eval = pd.concat([data_eval.reset_index(drop=True).drop(columns=['gender']), gender_encoded_df], axis=1)

    # Extracting features from the audio files
    data_eval = audio_feature_extraction(data_eval, num_segmentation)

    # return a cvs file with the new features
    data_eval.to_csv(f'datasets/evaluation_features_{num_segmentation}.csv', index = False)
    print(f'File saved as evaluation_features_{num_segmentation}_prova.csv')


# if __name__ == '__main__':
#     new_csv_with_audio_extracted_features() 