import torch
from torch import nn

from src.layers import DataEmbedding_wo_pos
from src.norm import Normalize


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DftSeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DftSeriesDecomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layer_num):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(down_sampling_layer_num)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high) # high resolution to low resolution
            out_low = out_low + out_low_res # add lowered high resolution to low resolution
            out_high = out_low # update high resolution
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2] # get next low resolution
            out_season_list.append(out_high.permute(0, 2, 1)) # append current added lower resolution to output list

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layer_num):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layer_num))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, pred_len, down_sampling_window, down_sampling_layer_num, d_model, channel_independence, decomp_method, moving_avg=None, top_k=None, d_ff=512, dropout=0.1):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = SeriesDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DftSeriesDecomp(top_k)
        else:
            raise ValueError('decompsition is error')

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len, down_sampling_window, down_sampling_layer_num)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len, down_sampling_window, down_sampling_layer_num)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.enc_in = 1  # Number of input features
        self.seq_len = 52
        self.pred_len = 12
        self.d_model = 64
        self.c_out = 1 # Number of output features
        self.down_sampling_window = 2
        self.down_sampling_layers_num = 2
        self.channel_independence = False
        self.e_layers = 3
        self.moving_avg = 5  # Example value for moving average
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layer_num=self.down_sampling_layers_num ,
            d_model=self.d_model,
            channel_independence=self.channel_independence,
            decomp_method='moving_avg',
            moving_avg=5,  # Example value for moving average
            top_k=None,  # Not used in moving_avg method
            d_ff=128,  # Feed-forward dimension
            dropout=0.1  # Dropout rate
        ) for _ in range(self.e_layers)])
        self.preprocess = SeriesDecomp(self.moving_avg)
        self.enc_in = 1

        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in,
                                                  d_model=self.d_model,
                                                  embed_type='fixed',
                                                  freq='w',
                                                  dropout=0.1)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=False)
                for _ in range(self.down_sampling_layers_num  + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len
                )
                for i in range(self.down_sampling_layers_num + 1)
            ]
        )

        self.projection_layer = nn.Linear(self.d_model, self.c_out, bias=True)

        self.out_res_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.seq_len // (self.down_sampling_window ** i),
                )
                for i in range(self.down_sampling_layers_num + 1)
            ]
        )

        self.regression_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.down_sampling_layers_num + 1)
            ]
        )

    def PreEncode(self, x_list):
        out1_list = []
        out2_list = []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)

        return (out1_list, out2_list)

    def MultiScaleProcessInputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)

        x_enc = x_enc.permute(0, 2, 1)  # Change to (batch_size, channels, seq_len)

        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0,2,1))

        for i in range(self.down_sampling_window_num):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0,2,1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def OutProjection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)

        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layer[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)

        dec_out = dec_out + out_res

        return dec_out

    def FutureMultiMixing(self, enc_out_list, x_list):
        dec_out_list = []
        for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]): # number, pdm_blocks, trend
            dec_out = self.predict_layers[i](enc_out.permute(0,2,1)).permute(0,2,1)
            dec_out = self.out_projection(dec_out, i, out_res)
            dec_out_list.append(dec_out)

        return dec_out_list

    def forecast(self, x_enc):
        x_enc = self.MultiScaleProcessInputs(x_enc)

        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x_list.append(x)

        enc_out_list = []
        x_list = self.PreEncode(x_list)
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            enc_out = self.enc_embedding(x) # only apply embedding to seasonal component
            enc_out_list.append(enc_out)

        for i in range(self.e_layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.FutureMultiMixing(B, enc_out_list, x_list[1])

        dec_out = torch.stack(dec_out_list, dim=1).sum(-1)  # Sum over the last dimension
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out