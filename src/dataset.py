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
    def __init__(self, price, fct:list=None, size=None, flag='train', train_pct=None, skip_col:list=None):
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

            d_stamp = pd.DataFrame(
                {
                    'month' : d.index.month,
                    'day' : d.index.day,
                }
            ).values
            d = d.values

            for j in range(0, len(d), self.seq_len):
                s_begin = j
                s_end = j + self.seq_len
                r_begin = s_end - self.label_len
                r_end = s_end + self.pred_len

                # break if not enough data for a full sequence
                if j + self.seq_len + self.pred_len > len(d):
                    break
                # set base for normalization
                base = d[j+self.seq_len-1].copy()
                if skip_col:
                    for col in skip_col:
                        base[col] = 1  # skip normalization for these columns
                # normalize data
                x = d[s_begin : s_end] / base
                y = d[r_begin : r_end] / base
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