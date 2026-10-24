import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features


class _ForecastDataset(Dataset):
    """Sliding (input, target) windows over one split of a CSV with a 'date' column.

    Values are standardized with statistics of the training split.
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        if size is None:
            size = [24 * 4 * 4, 24 * 4, 24 * 4]
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ["train", "test", "val"]
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _prepare(self, df_raw):
        return df_raw

    def _borders(self, n_rows):
        """Start and end row of the train, val and test splits."""
        raise NotImplementedError

    def _read_data(self):
        self.scaler = StandardScaler()
        df_raw = self._prepare(
            pd.read_csv(os.path.join(self.root_path, self.data_path))
        )

        border1s, border2s = self._borders(len(df_raw))
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            df_data = df_raw[df_raw.columns[1:]]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2].copy()
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class _ETTDataset(_ForecastDataset):
    """ETT: 12 months train, 4 months val, 4 months test."""

    steps_per_hour = 1

    def _borders(self, n_rows):
        month = 30 * 24 * self.steps_per_hour
        border1s = [0, 12 * month - self.seq_len, 12 * month + 4 * month - self.seq_len]
        border2s = [12 * month, 12 * month + 4 * month, 12 * month + 8 * month]
        return border1s, border2s


class Dataset_ETT_hour(_ETTDataset):
    steps_per_hour = 1


class Dataset_ETT_minute(_ETTDataset):
    steps_per_hour = 4


class Dataset_Custom(_ForecastDataset):
    """Any CSV with a 'date' column: 70% train, 10% val, 20% test."""

    def _prepare(self, df_raw):
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        return df_raw[["date"] + cols + [self.target]]

    def _borders(self, n_rows):
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_vali = n_rows - num_train - num_test
        border1s = [0, num_train - self.seq_len, n_rows - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n_rows]
        return border1s, border2s
