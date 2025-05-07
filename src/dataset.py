import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TS_dataset(Dataset):
    def __init__(self, price, fct=None, size=None, flag='train', train_pct=None):
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

        if train_pct is None:
            pct = [0.6, 0.1, 0.3]
        else:
            assert len(train_pct) == 3
            pct = train_pct
        brdst_l = [0, len(price)*pct[0] - self.seq_len, len(price)*pct[0]+len(price)*pct[1] - self.seq_len]
        brded_l = [len(price)*pct[0], len(price)*pct[0]+len(price)*pct[1], len(price)*pct[0]+len(price)*pct[1]+len(price)*pct[2]]
        brdst = int(brdst_l[self.type])
        brded = int(brded_l[self.type])
        price = price.iloc[brdst:brded]

        # make data pair
        data = []
        for i in range(len(price.columns)):
            p = price.iloc[:, i].dropna()
            if len(p) < self.seq_len + self.pred_len:
                continue
            if fct is not None:
                f = fct.loc[:p.index[-1]].dropna()
                d = pd.concat([p, f], axis=1).dropna().values
            else:
                d = p.values
            if len(d) < self.seq_len + self.pred_len:
                continue

            for j in range(0, len(d), self.seq_len):
                if j + self.seq_len + self.pred_len > len(d):
                    break
                x = d[j : j+self.seq_len] / d[j+self.seq_len]
                y = d[j+self.seq_len : j+self.seq_len+self.pred_len] / d[j+self.seq_len]
                data.append([x.reshape((self.seq_len, -1)), y.reshape((self.pred_len, -1))])

        self.data = data

    def __getitem__(self, index):
        x = torch.FloatTensor(self.data[index][0])
        y = torch.FloatTensor(self.data[index][1])

        return x, y

    def __len__(self):
        return len(self.data)