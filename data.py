from constants import *
from data_utils import load_json, build_sine_wave, modify_sine
from torch_data import SineWaveTorchDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np


class Dataset:
    def __init__(self, dataset_filepath: str = JSON_FILE_PATH):
        self.data_instruction = load_json(dataset_filepath)
        self.build_data()
        self.batch_size = None

    def build_data(self):
        self.build_data_features()
        dataset_size = self.data_instruction['dataset_size']
        num_points = self.data_instruction['num_points']
        self.X = np.zeros((dataset_size, num_points))
        self.Y = np.zeros(dataset_size, dtype=int)

        ratio = self.data_instruction['normal_anomalous_ratio']

        for idx in range(dataset_size):
            time_steps = np.linspace(0, 2 * np.pi, num_points)
            freq = self.features_dict['freq'][idx]
            amp = self.features_dict['amp'][idx]
            loc = self.features_dict['loc'][idx]
            length = self.features_dict['length'][idx]

            label = np.random.choice([0, 1], p=[ratio, 1 - ratio])
            wave = build_sine_wave(time_steps, freq, amp)

            if label == 1:
                wave = modify_sine(wave, loc, length)
            self.X[idx] = wave
            self.Y[idx] = label

    def build_data_key_features(self, key: str = 'loc') -> np.ndarray:
        """
        Generates feature values from the specified JSON parameters.
        """
        if key in ['loc', 'length']:
            min_val = int(self.data_instruction[f'min_{key}'] * self.data_instruction['num_points'])
            max_val = int(self.data_instruction[f'max_{key}'] * self.data_instruction['num_points'])
            step = max(1, int(self.data_instruction[f'step_{key}'] * self.data_instruction['num_points']))
        else:
            min_val = self.data_instruction[f'min_{key}']
            max_val = self.data_instruction[f'max_{key}']
            step = self.data_instruction[f'step_{key}']

        values = np.arange(min_val, max_val, step)
        return np.random.choice(values, size=self.data_instruction['dataset_size'])

    def build_data_features(self):
        self.features_dict = {}
        for key in KEY_LIST:
            self.features_dict[key] = self.build_data_key_features(key)

    def to_torch(self, batch_size: int = DEFAULT_BATCH_SIZE) -> SineWaveTorchDataset:
        self.batch_size = batch_size
        self.torch_data = SineWaveTorchDataset(self.X, self.Y)
        return self.torch_data

    def train_test_split(self, test_size: float = 0.2, torch_data: bool = True):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=42
        )

        if torch_data:
            self.train_torch_data = SineWaveTorchDataset(self.X_train, self.Y_train)
            self.test_torch_data = SineWaveTorchDataset(self.X_test, self.Y_test)

            if self.batch_size is None:
                self.batch_size = DEFAULT_BATCH_SIZE

            self.train_loader = DataLoader(self.train_torch_data, batch_size=self.batch_size, shuffle=True)
            self.test_loader = DataLoader(self.test_torch_data, batch_size=self.batch_size, shuffle=False)

    def train_val_test_split(self, val_size: float = 0.1, test_size: float = 0.2, torch_data: bool = True, batch_size: int = None):
        """
        Splits data into train, validation, and test sets, with optional torch conversion.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = DEFAULT_BATCH_SIZE

        # Step 1: Train+Val and Test split
        X_trainval, X_test, Y_trainval, Y_test = train_test_split(
            self.X, self.Y, test_size=test_size, random_state=42
        )

        # Step 2: Train and Validation split
        val_adjusted = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_trainval, Y_trainval, test_size=val_adjusted, random_state=42
        )

        # Store splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.Y_train, self.Y_val, self.Y_test = Y_train, Y_val, Y_test

        if torch_data:
            self.train_torch_data = SineWaveTorchDataset(X_train, Y_train)
            self.val_torch_data = SineWaveTorchDataset(X_val, Y_val)
            self.test_torch_data = SineWaveTorchDataset(X_test, Y_test)

            self.train_loader = DataLoader(self.train_torch_data, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(self.val_torch_data, batch_size=self.batch_size, shuffle=False)
            self.test_loader = DataLoader(self.test_torch_data, batch_size=self.batch_size, shuffle=False)
