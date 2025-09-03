import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"✅ Val loss improved to {val_loss:.6f}. Resetting counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t), Recurrent_block(ch_out, t=t)
        )

    def forward(self, x):
        x = self.Conv_1x1(x)
        return x + self.RCNN(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, t=2, base_filters=64, depth=4):
        super(R2AttU_Net, self).__init__()

        self.depth = depth
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder = nn.ModuleList()
        self.decoder_up = nn.ModuleList()
        self.att_blocks = nn.ModuleList()
        self.decoder_rrcnn = nn.ModuleList()

        # Encoder
        in_ch = img_ch
        for i in range(depth):
            out_ch = base_filters * (2**i)
            self.encoder.append(RRCNN_block(ch_in=in_ch, ch_out=out_ch, t=t))
            in_ch = out_ch

        # Decoder
        for i in reversed(range(depth - 1)):
            ch_in = base_filters * (2 ** (i + 1))
            ch_out = base_filters * (2**i)

            self.decoder_up.append(up_conv(ch_in=ch_in, ch_out=ch_out))
            self.att_blocks.append(
                Attention_block(F_g=ch_out, F_l=ch_out, F_int=ch_out // 2)
            )
            self.decoder_rrcnn.append(RRCNN_block(ch_in=ch_out * 2, ch_out=ch_out, t=t))

        # Final 1x1 conv
        self.Conv_1x1 = nn.Conv2d(
            base_filters, output_ch, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        enc_features = []

        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)
            x = self.Maxpool(x)

        x = enc_features.pop()  # deepest layer output (after all MaxPool)

        for i in range(self.depth - 1):
            x = self.decoder_up[i](x)
            skip = enc_features.pop()
            skip = self.att_blocks[i](x, skip)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_rrcnn[i](x)

        x = self.Conv_1x1(x)
        return x
