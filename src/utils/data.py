import pandas as pd
import jax.numpy as jnp
import numpy as np
from jax import random
from sklearn.preprocessing import MinMaxScaler 


class TimeSeriesLoader:
    def __init__(self, config):
        self.config = config
        self.window_size = config.window_size
        self.horizon = config.horizon  
        self.batch_size = config.batch_size
        self.shuffle = getattr(config, "shuffle", False)
        self.split_ratio = getattr(config, "split_ratio", 0.7)
        self.start_date = pd.to_datetime(config.start_date, dayfirst=True)
        self.end_date = pd.to_datetime(config.end_date, dayfirst=True)
        self.filename = f"{config.raw_data_dir}/{config.dataset.upper()}"

        self.df = self._read_csv(self.filename, self.start_date, self.end_date)
        # self.df = self._fill_gaps(self.df, self.start_date, self.end_date)
        self.df = self._normalize_data(self.df)
        self.df.replace([np.inf, -np.inf], np.nan,
                        inplace=True)  
        self.df.bfill(inplace=True)
        self.X, self.y = self._create_sequences(
            self.df.to_numpy()
        )  
        self._create_splits()

    def _read_csv(self, pth, start_date, end_date):
        print(f"Reading File From {pth}")
        df = pd.read_csv(
            pth + ".csv", parse_dates=["Date"], index_col="Date")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        if df.index.max() < start_date or df.index.min() > end_date:
            df.dropna(inplace=True)
            return pd.DataFrame()
        df = df[["Open", "Close", "Low", "High"]]
        return df.loc[start_date:end_date]

    def _fill_gaps(self, df, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        df = df.reindex(dates).ffill().bfill()
        df.index.name = "Date"
        return df

    def _normalize_data(self, stock_data):
        self.scaler = MinMaxScaler()  
        df_normalized = pd.DataFrame(
            self.scaler.fit_transform(stock_data),
            columns=stock_data.columns,
            index=stock_data.index
        )
        if self.config.task == "return":
            df_normalized["daily-return"] = np.log(df_normalized["Close"] / df_normalized["Close"].shift(1))
        return df_normalized

    def _create_sequences(self, data):
        num_timesteps, _ = data.shape
        total_window = self.window_size + self.horizon

        num_sequences = num_timesteps - total_window + 1
        indices = jnp.arange(total_window) + jnp.arange(num_sequences)[:, None]
        sequences = data[indices]
        X = sequences[:, :self.window_size, :]   

        if self.config.task == "return":
            y = sequences[:, -self.horizon:, -1]
        else:
            y = sequences[:, -self.horizon:, 1]
        return X, y

    def _create_splits(self):
        n = len(self.X)
        split1 = int(n * self.split_ratio)
        split2 = int(n * (self.split_ratio + (1 - self.split_ratio) / 2))

        self.X_train, self.y_train = self.X[:split1], self.y[:split1]
        self.X_calib, self.y_calib = self.X[split1:split2], self.y[split1:split2]
        self.X_test, self.y_test = self.X[split2:], self.y[split2:]

    def get_loader(self, data_split="train", key=None):
        """Get JAX-compatible data loader that yields batches."""
        X, y = {
            "train": (self.X_train, self.y_train),
            "calib": (self.X_calib, self.y_calib),
            "test": (self.X_test, self.y_test),
        }[data_split]

        n = X.shape[0]
        indices = jnp.arange(n)

        if self.shuffle and key is not None:
            indices = random.permutation(key, indices)

        num_batches = n // self.batch_size
        for i in range(num_batches):
            X_batch = X[indices[i *
                                self.batch_size: (i + 1) * self.batch_size]]
            y_batch = y[indices[i *
                                self.batch_size: (i + 1) * self.batch_size]]

            X_batch = X_batch.reshape(self.batch_size, self.window_size, -1)
            y_batch = y_batch.reshape(self.batch_size, self.horizon)

            yield X_batch, y_batch

    def unnormalize(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
