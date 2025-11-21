import warnings

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset
from tqdm import tqdm
# ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TS_dataset(Dataset):
    def __init__(self, price, fct:list=None, size=None, flag='train', train_pct=None, std_scale:bool=False, skip_col:list=None, port_weight:pd.DataFrame=None):
        if size is None:
            self.seq_len = 52
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'valid']
        type_map = {
            'train': 0,
            'test': 1,
            'valid': 2
        }
        self.type = type_map[flag]
        self.scaler = StandardScaler()

        if train_pct is None:
            pct = [0.6, 0.2, 0.2]
        else:
            assert len(train_pct) == 3
            pct = train_pct

        def mark_non_null_count(df, n):
            total_non_nulls = (~df.isna()).cumsum() - 1  # 이전까지 비결측 개수

            # total_non_nulls가 n 이상이면 1, 아니면 0
            result = (total_non_nulls >= n).astype(int)
            return result

        df_result = mark_non_null_count(price, self.seq_len + self.pred_len)

        # should split dataset based on ratio of row * column numbers
        price_len = df_result.sum(axis=1); rolling_len = price_len.cumsum(); total_len = price_len.sum()
        train_num = (rolling_len <= total_len * pct[0]).sum()
        valid_num = (rolling_len <= total_len * (pct[0] + pct[1])).sum()
        test_num = (rolling_len <= total_len).sum()

        brdst_l = [0, train_num - self.seq_len, valid_num - self.seq_len]
        brded_l = [train_num, valid_num, test_num]
        brdst = int(brdst_l[self.type])
        brded = int(brded_l[self.type])
        price = price.iloc[brdst:brded]

        # make data pair
        data = []
        data_stamp = []
        for i in tqdm(range(len(price.columns))):
            p = price.iloc[:, i].dropna()
            if len(p) < self.seq_len + self.pred_len:
                continue
            if fct:
                d = []
                for f in fct:
                    try:
                        f_partial = f.loc[:p.index[-1], p.name].dropna().reindex(p.index, method='ffill')
                    except:
                        f_partial = pd.Series()
                    if len(f_partial)==0:
                        f_partial = pd.Series([0]*len(p), index=p.index, name=p.name)
                    d.append(f_partial)
                d.append(p) # append target series to the end
                d = pd.concat(d, axis=1).dropna()
            else:
                d = p
            if len(d) < self.seq_len + self.pred_len:
                continue

            d_name = p.name
            d_timestamp = p.index
            d_stamp = pd.DataFrame(
                {
                    'month' : d_timestamp.month,
                    'day' : d_timestamp.day,
                }
            ).values
            d = d.values

            for j in range(0, len(d) - self.seq_len - self.pred_len + 1, self.pred_len):
                s_begin = j
                s_end = j + self.seq_len
                r_begin = s_end - self.label_len
                r_end = s_end + self.pred_len

                x_base = d[s_begin: s_end]
                y_base = d[r_begin: r_end]

                if port_weight:
                    try:
                        i_date_loc = port_weight.index.get_indexer([d_timestamp[s_end]], method='nearest')[0]
                        i_loc = port_weight.columns.get_loc(d_name)
                        if port_weight.iloc[i_date_loc, i_loc] == 1:
                            pass
                    except:
                        continue

                # normalize target data first
                base = d[s_end - 1].copy()
                x_norm = x_base / base
                y_norm = y_base / base
                x_base[:, -1] = x_norm[:, -1]  # last column should be normalized
                y_base[:, -1] = y_norm[:, -1]  # last column should be normalized

                # normalize data
                if std_scale:
                    x = self.scaler.fit_transform(x_base)
                    y = self.scaler.transform(y_base)
                else:
                    x = x_norm.copy()
                    y = y_norm.copy()

                # Restore specific columns if needed
                if skip_col:
                    x[:, skip_col] = x_base[:, skip_col]
                    y[:, skip_col] = y_base[:, skip_col]

                # clip values to avoid overflow, preserve data for last column
                x = x.clip(-3, 3)
                y = y.clip(-3, 3)

                data.append([x, y])
                # append timestamps
                x_stamp = d_stamp[s_begin : s_end]
                y_stamp = d_stamp[r_begin : r_end]
                data_stamp.append([x_stamp, y_stamp])

        self.data = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index][0])
        y = torch.FloatTensor(self.data[index][1])
        x_mark = torch.FloatTensor(self.data_stamp[index][0])
        y_mark = torch.FloatTensor(self.data_stamp[index][1])

        return x, y, x_mark, y_mark

    def __len__(self):
        return len(self.data)
