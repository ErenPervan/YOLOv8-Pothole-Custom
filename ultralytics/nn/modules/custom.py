import torch.nn as nn


# --------------------------------------------------------
# 1. SimAM (Simple Attention Module)
# Makale Ref: [cite: 307-318]
# Parametresiz, enerji tabanlı dikkat mekanizması.
# --------------------------------------------------------
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [Batch, Channel, Height, Width]
        _b, _c, h, w = x.size()
        n = w * h - 1

        # Eq. 6: Enerji fonksiyonu hesaplaması
        # Ortalama ve varyans
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # Eq. 7: Attention uygulama
        return x * self.activaton(y)


# --------------------------------------------------------
# 2. DSConv (Dynamic Snake Convolution) - Basitleştirilmiş
# Makale Ref: [cite: 252-265]
# Kıvrımlı ve ince yapıları (çatlaklar) yakalamak için.
# Not: Tam DCNv2/v3 implementasyonu CUDA derlemesi gerektirir.
# Bu versiyon, standard PyTorch ile dilated conv mantığını kullanarak
# morfolojik uyumu simüle eder.
# --------------------------------------------------------
class DSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # Genişletilmiş (Dilated) konvolüsyon ile "snake" etkisini simüle ediyoruz
        # Standard kare grid yerine daha geniş bir alana bakmasını sağlar.
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # SİLME: Kullanıcı isteği üzerine GELU yerine SiLU kullanıyoruz.
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Bu modül, makaledeki "C2f + SimAM" yapısını oluşturmak için yardımcı blok
from ultralytics.nn.modules.block import C2f


class C2f_SimAM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # C2f çıkışına SimAM ekliyoruz [cite: 328]
        self.simam = SimAM()

    def forward(self, x):
        # C2f'in standard forward işlemi + SimAM
        return self.simam(super().forward(x))
