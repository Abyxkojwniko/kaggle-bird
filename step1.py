#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
import joblib, json, re

from  sklearn.model_selection  import StratifiedKFold
tqdm.pandas()


# In[2]:


df = pd.read_csv('/root/autodl-tmp/train_metadata.csv')
df['secondary_labels'] = df['secondary_labels'].apply(lambda x: re.findall(r"'(\w+)'", x))
df['len_sec_labels'] = df['secondary_labels'].map(len)


# In[3]:


from sklearn.model_selection import train_test_split
import pandas as pd

def birds_stratified_split(df, target_col, test_size=0.2):
    class_counts = df[target_col].value_counts()
    low_count_classes = class_counts[class_counts < 2].index.tolist() ### Birds with single counts

    df['train'] = df[target_col].isin(low_count_classes)

    train_df, val_df = train_test_split(df[~df['train']], test_size=test_size, stratify=df[~df['train']][target_col], random_state=42)

    train_df = pd.concat([train_df, df[df['train']]], axis=0).reset_index(drop=True)

    # Remove the 'valid' column
    train_df.drop('train', axis=1, inplace=True)
    val_df.drop('train', axis=1, inplace=True)

    return train_df, val_df


# In[4]:


train_df, valid_df = birds_stratified_split(df, 'primary_label', 0.2)


# In[5]:


class Config:
    sampling_rate = 32000
    duration = 5 
    fmin = 0
    fmax = None
    audios_path = Path("/root/autodl-tmp/train_audio")
    out_dir_train = Path("specs/train") 
    
    out_dir_valid = Path("specs/valid") 


# In[6]:


Config.out_dir_train.mkdir(exist_ok=True, parents=True)
Config.out_dir_valid.mkdir(exist_ok=True, parents=True)


# In[7]:


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}


# In[8]:


def add_path_df(df):
    
    df["path"] = [str(Config.audios_path/filename) for filename in df.filename]
    df = df.reset_index(drop=True)
    pool = joblib.Parallel(2)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.path]
    df2 =  pd.DataFrame(pool(tqdm(tasks))).reset_index(drop=True)
    df = pd.concat([df,df2], axis=1).reset_index(drop=True)

    return df


# In[9]:


tqdm.pandas()


# In[10]:


train_df = add_path_df(train_df)
valid_df = add_path_df(valid_df)


# In[11]:


def compute_melspec(y, sr, n_mels, fmin, fmax):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = lb.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    melspec = lb.power_to_db(melspec).astype(np.float32)
    return melspec


# In[12]:


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)
    
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V

def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        
        n_repeats = length // len(y)
        epsilon = length % len(y)
        
        y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y
 


# In[15]:


class AudioToImage:
    def __init__(self, sr=Config.sampling_rate, n_mels=128, fmin=Config.fmin, fmax=Config.fmax, duration=Config.duration, step=None, res_type="kaiser_fast", resample=True, train = True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.train = train
    def audio_to_image(self, audio):
        melspec = compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax ) 
        image = mono_to_color(melspec)
#         compute_melspec(y, sr, n_mels, fmin, fmax)
        return image

    def __call__(self, row, save=True):

        audio, orig_sr = sf.read(row.path, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
        
        audios = [audio[i:i+self.audio_length] for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1] , length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)
        
        if save:
            if self.train:
                path = Config.out_dir_train/f"{row.filename}.npy"
            else:
                path = Config.out_dir_valid/f"{row.filename}.npy"
            
            path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(path), images)
        else:
            return  row.filename, images


# In[16]:


tqdm.pandas()


# In[17]:


def get_audios_as_images(df, train = True):
    pool = joblib.Parallel(2)
    
    converter = AudioToImage(step=int(Config.duration*0.666*Config.sampling_rate),train=train)
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]
    pool(tqdm(tasks))


# In[18]:


get_audios_as_images(train_df, train = True)
get_audios_as_images(valid_df, train = False)


# In[19]:


train_df.to_csv('/root/train.csv',index=False)
valid_df.to_csv('/root/valid.csv',index=False)


# In[ ]:




