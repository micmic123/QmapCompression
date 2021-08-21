import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperpriors import ScaleHyperprior
from .utils import conv
from .layers import GDN1

from .layers import Conv2d, UpConv2d, SFT, SFTResblk


class Ablation_wo_SFT(ScaleHyperprior):
    def __init__(self, N=192, M=192, prior_nc=64, **kwargs):
        super().__init__(N, M, **kwargs)
        # featured qmap
        self.qmap_feature_g1 = nn.Sequential(
            conv(4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g5 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_h3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # for decoder
        self.qmap_feature_gs0 = nn.Sequential(
            UpConv2d(N, N // 2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N // 2, N // 4),
            nn.LeakyReLU(0.1, True),
            conv(N // 4, N // 4, 3, 1)
        )
        self.qmap_feature_gs1 = nn.Sequential(
            conv(N + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs5 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        self.g_a = None
        self.g_a0 = Conv2d(3 + prior_nc, N // 4, stride=1, kernel_size=5)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = Conv2d(N // 4 + prior_nc, N // 2)
        self.g_a3 = GDN1(N//2)
        self.g_a4 = Conv2d(N // 2 + prior_nc, N)
        self.g_a5 = GDN1(N)
        self.g_a6 = Conv2d(N + prior_nc, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = Conv2d(N + prior_nc, M)

        self.g_s = None
        self.g_s0 = UpConv2d(M + prior_nc, N)
        self.g_s1 = GDN1(N, inverse=True)
        self.g_s2 = UpConv2d(N + prior_nc, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = UpConv2d(N + prior_nc, N // 2)
        self.g_s5 = GDN1(N//2, inverse=True)
        self.g_s6 = UpConv2d(N // 2 + prior_nc, N // 4)
        self.g_s7 = GDN1(N//4, inverse=True)
        self.g_s8 = Conv2d(N // 4 + prior_nc, 3, stride=1, kernel_size=5)

        self.h_a = None
        self.h_a0 = Conv2d(M + prior_nc, N, stride=1, kernel_size=3)
        self.h_a1 = nn.LeakyReLU(inplace=True)
        self.h_a2 = Conv2d(N + prior_nc, N)
        self.h_a3 = nn.LeakyReLU(inplace=True)
        self.h_a4 = Conv2d(N + prior_nc, N)

        # hyperprior decoder
        self.h_s = nn.Sequential(
            UpConv2d(N, M),
            nn.LeakyReLU(inplace=True),
            UpConv2d(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def _down_like(self, x, target):
        return F.adaptive_avg_pool2d(x, target.size()[2:])

    def _down(self, x, scale=16):
        return F.adaptive_avg_pool2d(x, (x.size(2)//scale, x.size(3)//scale))

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(torch.cat([qmap, x], dim=1))
        x = self.g_a0(torch.cat([qmap, x], dim=1))
        x = self.g_a1(x)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a2(torch.cat([qmap, x], dim=1))
        x = self.g_a3(x)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a4(torch.cat([qmap, x], dim=1))
        x = self.g_a5(x)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a6(torch.cat([qmap, x], dim=1))
        x = self.g_a7(x)

        qmap = self.qmap_feature_g5(qmap)
        x = self.g_a8(torch.cat([qmap, x], dim=1))

        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(torch.cat([w, x], dim=1))
        x = self.g_s1(x)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(torch.cat([w, x], dim=1))
        x = self.g_s3(x)

        w = self.qmap_feature_gs3(w)
        x = self.g_s4(torch.cat([w, x], dim=1))
        x = self.g_s5(x)

        w = self.qmap_feature_gs4(w)
        x = self.g_s6(torch.cat([w, x], dim=1))
        x = self.g_s7(x)

        w = self.qmap_feature_gs5(w)
        x = self.g_s8(torch.cat([w, x], dim=1))

        return x

    def h_a(self, x, qmap):
        qmap = self.qmap_feature_h1(torch.cat([self._down_like(qmap, x), x], dim=1))
        x = self.h_a0(torch.cat([qmap, x], dim=1))
        x = self.h_a1(x)

        qmap = self.qmap_feature_h2(qmap)
        x = self.h_a2(torch.cat([qmap, x], dim=1))
        x = self.h_a3(x)

        qmap = self.qmap_feature_h3(qmap)
        x = self.h_a4(torch.cat([qmap, x], dim=1))

        return x


    def forward(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class Ablation_wo_SC(ScaleHyperprior):
    def __init__(self, N=192, M=192, sft_ks=3, prior_nc=64, **kwargs):
        super().__init__(N, M, **kwargs)
        # featured qmap
        self.qmap_feature_g1 = nn.Sequential(
            conv(1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_g2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g4 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_g5 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_h1 = nn.Sequential(
            conv(1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_h3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # for decoder
        self.qmap_feature_gs0 = nn.Sequential(
            UpConv2d(N, N//2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N//2, N//4),
            nn.LeakyReLU(0.1, True),
            conv(N//4, N//4, 3, 1)
        )
        self.qmap_feature_gs1 = nn.Sequential(
            conv(N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs4 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )
        self.qmap_feature_gs5 = nn.Sequential(
            UpConv2d(prior_nc, prior_nc, 3),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc, prior_nc, 1, 1)
        )

        # main encoder
        self.g_a = None
        self.g_a0 = Conv2d(3, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_a3 = Conv2d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv2d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a9 = Conv2d(N, N)
        self.g_a10 = GDN1(N)
        self.g_a11 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv2d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # hyperprior encoder
        self.h_a = None
        self.h_a0 = Conv2d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        self.h_a3 = Conv2d(N, N)
        self.h_a4 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a5 = nn.LeakyReLU(inplace=True)

        self.h_a6 = Conv2d(N, N)
        self.h_a7 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a8 = SFTResblk(N, prior_nc, ks=sft_ks)

        # main decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv2d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv2d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv2d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s11 = UpConv2d(N // 2, N // 4)
        self.g_s12 = GDN1(N // 4, inverse=True)
        self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv2d(N // 4, 3, kernel_size=5, stride=1)

        # hyperprior decoder
        self.h_s = nn.Sequential(
            UpConv2d(N, M),
            nn.LeakyReLU(inplace=True),
            UpConv2d(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


    def _down_like(self, x, target):
        return F.adaptive_avg_pool2d(x, target.size()[2:])

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        qmap = self.qmap_feature_g2(qmap)
        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        qmap = self.qmap_feature_g3(qmap)
        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        qmap = self.qmap_feature_g4(qmap)
        x = self.g_a9(x)
        x = self.g_a10(x)
        x = self.g_a11(x, qmap)

        qmap = self.qmap_feature_g5(qmap)
        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        qmap = self.qmap_feature_h1(self._down_like(qmap, x))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        qmap = self.qmap_feature_h2(qmap)
        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x)

        qmap = self.qmap_feature_h3(qmap)
        x = self.h_a6(x)
        x = self.h_a7(x, qmap)
        x = self.h_a8(x, qmap)
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(w)
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        w = self.qmap_feature_gs4(w)
        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        w = self.qmap_feature_gs5(w)
        x = self.g_s11(x)
        x = self.g_s12(x)
        x = self.g_s13(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class Ablation_wo_CS(ScaleHyperprior):
    def __init__(self, N=192, M=192, sft_ks=3, prior_nc=64, **kwargs):
        super().__init__(N, M, **kwargs)
        # featured qmap
        self.qmap_feature_g1 = nn.Sequential(
            conv(4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h1 = nn.Sequential(
            conv(M + 1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        # for decoder
        self.qmap_feature_gs0 = nn.Sequential(
            UpConv2d(N, N//2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N//2, N//4),
            nn.LeakyReLU(0.1, True),
            conv(N//4, N//4, 3, 1)
        )
        self.qmap_feature_gs1 = nn.Sequential(
            conv(N + N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )

        # main encoder
        self.g_a = None
        self.g_a0 = Conv2d(3, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_a3 = Conv2d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_a6 = Conv2d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a9 = Conv2d(N, N)
        self.g_a10 = GDN1(N)
        self.g_a11 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv2d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # hyperprior encoder
        self.h_a = None
        self.h_a0 = Conv2d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        self.h_a3 = Conv2d(N, N)
        self.h_a4 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a5 = nn.LeakyReLU(inplace=True)

        self.h_a6 = Conv2d(N, N)
        self.h_a7 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a8 = SFTResblk(N, prior_nc, ks=sft_ks)

        # main decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv2d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv2d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv2d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s11 = UpConv2d(N // 2, N // 4)
        self.g_s12 = GDN1(N // 4, inverse=True)
        self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv2d(N // 4, 3, kernel_size=5, stride=1)

        # hyperprior decoder
        self.h_s = nn.Sequential(
            UpConv2d(N, M),
            nn.LeakyReLU(inplace=True),
            UpConv2d(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


    def _down_like(self, x, target):
        return F.adaptive_avg_pool2d(x, target.size()[2:])

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(torch.cat([qmap, x], dim=1))
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        x = self.g_a9(x)
        x = self.g_a10(x)
        x = self.g_a11(x, qmap)

        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        qmap = self.qmap_feature_h1(torch.cat([self._down_like(qmap, x), x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x)

        x = self.h_a6(x)
        x = self.h_a7(x, qmap)
        x = self.h_a8(x, qmap)
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(torch.cat([w, x], dim=1))
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        x = self.g_s11(x)
        x = self.g_s12(x)
        x = self.g_s13(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class Ablation_wo_CS_SC(ScaleHyperprior):
    def __init__(self, N=192, M=192, sft_ks=3, prior_nc=64, **kwargs):
        super().__init__(N, M, **kwargs)
        # featured qmap
        self.qmap_feature_g1 = nn.Sequential(
            conv(1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        self.qmap_feature_h1 = nn.Sequential(
            conv(1, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )
        # for decoder
        self.qmap_feature_gs0 = nn.Sequential(
            UpConv2d(N, N//2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N//2, N//4),
            nn.LeakyReLU(0.1, True),
            conv(N//4, N//4, 3, 1)
        )
        self.qmap_feature_gs1 = nn.Sequential(
            conv(N // 4, prior_nc * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 4, prior_nc * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(prior_nc * 2, prior_nc, 3, 1)
        )

        # main encoder
        self.g_a = None
        self.g_a0 = Conv2d(3, N//4, kernel_size=5, stride=1)
        self.g_a1 = GDN1(N//4)
        self.g_a2 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_a3 = Conv2d(N//4, N//2)
        self.g_a4 = GDN1(N//2)
        self.g_a5 = SFT(N // 2, prior_nc, ks=sft_ks)
        self.g_a6 = Conv2d(N//2, N)
        self.g_a7 = GDN1(N)
        self.g_a8 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a9 = Conv2d(N, N)
        self.g_a10 = GDN1(N)
        self.g_a11 = SFT(N, prior_nc, ks=sft_ks)

        self.g_a12 = Conv2d(N, M)
        self.g_a13 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_a14 = SFTResblk(M, prior_nc, ks=sft_ks)

        # hyperprior encoder
        self.h_a = None
        self.h_a0 = Conv2d(M, N, kernel_size=3, stride=1)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.LeakyReLU(inplace=True)

        self.h_a3 = Conv2d(N, N)
        self.h_a4 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a5 = nn.LeakyReLU(inplace=True)

        self.h_a6 = Conv2d(N, N)
        self.h_a7 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a8 = SFTResblk(N, prior_nc, ks=sft_ks)

        # main decoder
        self.g_s = None
        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = UpConv2d(M, N)
        self.g_s3 = GDN1(N, inverse=True)
        self.g_s4 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s5 = UpConv2d(N, N)
        self.g_s6 = GDN1(N, inverse=True)
        self.g_s7 = SFT(N, prior_nc, ks=sft_ks)

        self.g_s8 = UpConv2d(N, N // 2)
        self.g_s9 = GDN1(N // 2, inverse=True)
        self.g_s10 = SFT(N // 2, prior_nc, ks=sft_ks)

        self.g_s11 = UpConv2d(N // 2, N // 4)
        self.g_s12 = GDN1(N // 4, inverse=True)
        self.g_s13 = SFT(N // 4, prior_nc, ks=sft_ks)

        self.g_s14 = Conv2d(N // 4, 3, kernel_size=5, stride=1)

        # hyperprior decoder
        self.h_s = nn.Sequential(
            UpConv2d(N, M),
            nn.LeakyReLU(inplace=True),
            UpConv2d(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )


    def _down_like(self, x, target):
        return F.adaptive_avg_pool2d(x, target.size()[2:])

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_g1(qmap)
        x = self.g_a0(x)
        x = self.g_a1(x)
        x = self.g_a2(x, qmap)

        x = self.g_a3(x)
        x = self.g_a4(x)
        x = self.g_a5(x, qmap)

        x = self.g_a6(x)
        x = self.g_a7(x)
        x = self.g_a8(x, qmap)

        x = self.g_a9(x)
        x = self.g_a10(x)
        x = self.g_a11(x, qmap)

        x = self.g_a12(x)
        x = self.g_a13(x, qmap)
        x = self.g_a14(x, qmap)
        return x

    def h_a(self, x, qmap):
        qmap = self.qmap_feature_h1(self._down_like(qmap, x))
        x = self.h_a0(x)
        x = self.h_a1(x, qmap)
        x = self.h_a2(x)

        x = self.h_a3(x)
        x = self.h_a4(x, qmap)
        x = self.h_a5(x)

        x = self.h_a6(x)
        x = self.h_a7(x, qmap)
        x = self.h_a8(x, qmap)
        return x

    def g_s(self, x, z):
        w = self.qmap_feature_gs0(z)
        w = self.qmap_feature_gs1(w)
        x = self.g_s0(x, w)
        x = self.g_s1(x, w)

        x = self.g_s2(x)
        x = self.g_s3(x)
        x = self.g_s4(x, w)

        x = self.g_s5(x)
        x = self.g_s6(x)
        x = self.g_s7(x, w)

        x = self.g_s8(x)
        x = self.g_s9(x)
        x = self.g_s10(x, w)

        x = self.g_s11(x)
        x = self.g_s12(x)
        x = self.g_s13(x, w)

        x = self.g_s14(x)
        return x

    def forward(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, qmap):
        y = self.g_a(x, qmap)
        z = self.h_a(y, qmap)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
