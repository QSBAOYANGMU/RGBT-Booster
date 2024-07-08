import torch
import torch.nn as nn

from utils.tensor_ops import cus_sample, upsample_add
from backbone.convnext import (LayerNorm, convnext_base,convnext_tiny)
from backbone.VGG import (
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
from module.pretrainteaher import (teacher_T, teacher_R)
from module.MyModules import (
    EDFM,
    EDFMT,
    IDEM,
    FDM,
    GLlocal,
    Graph_Attention_Union,
    PE,
    HF,
    MS,
    # IDEM_1,
    IDEMSE,
    Enhanced_P,
    # Glocal,
    Fusion,
)
import warnings
warnings.filterwarnings("ignore")

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class Teacher(nn.Module):
    def __init__(self, pretrained=True):
        super(Teacher, self).__init__()
        self.teacher_r = teacher_R(pretrained=True)
        for p in self.teacher_r.parameters():
            p.requires_grad = False
        self.teacher_t=teacher_T(pretrained=True)
        for p in self.teacher_t.parameters():
            p.requires_grad = False
    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]
        rgbt = [in_depth, in_data]
        layer_r, teacher_r = self.teacher_r(rgbt)
        layer_t, teacher_t = self.teacher_t(RGBT)
        return  layer_r, teacher_r , layer_t, teacher_t

class DEFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DEFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.vgg_r = convnext_tiny(pretrained=True, in_22k=True)
        self.vgg_d = convnext_tiny(pretrained=True, in_22k=True)##96, 192, 384, 768
        self.vgg_r_detail = convnext_base(pretrained=True, in_22k=True)
        self.vgg_d_detail = convnext_base(pretrained=True, in_22k=True)  ##96, 192, 384, 768
        self.layer0 = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1 = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2 = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3 = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layerd0 = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layerd1 = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layerd2 = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layerd3 = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_detail = nn.Sequential(self.vgg_r_detail.downsample_layers[0], self.vgg_r_detail.stages[0],
                                        LayerNorm(128, eps=1e-6, data_format="channels_first"))
        self.layer1_detail = nn.Sequential(self.vgg_r_detail.downsample_layers[1], self.vgg_r_detail.stages[1],
                                        LayerNorm(256, eps=1e-6, data_format="channels_first"))
        # self.layer2_detail = nn.Sequential(self.vgg_r_detail.downsample_layers[2], self.vgg_r_detail.stages[2],
        #                                 LayerNorm(512, eps=1e-6, data_format="channels_first"))
        # self.layer3_detail = nn.Sequential(self.vgg_r_detail.downsample_layers[3], self.vgg_r_detail.stages[3],
        #                                 LayerNorm(1024, eps=1e-6, data_format="channels_first"))

        self.layerd0_detail  = nn.Sequential(self.vgg_r_detail.downsample_layers[0], self.vgg_r_detail.stages[0],
                                         LayerNorm(128, eps=1e-6, data_format="channels_first"))
        self.layerd1_detail  = nn.Sequential(self.vgg_r_detail.downsample_layers[1], self.vgg_r_detail.stages[1],
                                         LayerNorm(256, eps=1e-6, data_format="channels_first"))
        # self.layerd2_detail  = nn.Sequential(self.vgg_r_detail.downsample_layers[2], self.vgg_r_detail.stages[2],
        #                                  LayerNorm(512, eps=1e-6, data_format="channels_first"))
        # self.layerd3_detail  = nn.Sequential(self.vgg_r_detail.downsample_layers[3], self.vgg_r_detail.stages[3],
        #                                  LayerNorm(1024, eps=1e-6, data_format="channels_first"))

        # self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(768, 64, 1)
        self.trans4 = nn.Conv2d(384, 64, 1)
        self.trans2 = nn.Conv2d(192, 64, 1)
        self.trans1 = nn.Conv2d(96, 32, 1)

        # self.t_trans16 = IDEM(512, 64)
        self.t_trans8 = IDEMSE(768, 64)
        self.t_trans4 = IDEMSE(384, 64)
        self.t_trans2 = IDEM(192,32)
        self.t_trans1 = IDEM(96,64)
        # self.t_trans8 = GLlocal(768, 64)
        # self.t_trans4 = GLlocal(384, 64)
        self.t_tt2 = GLlocal(192,32)
        self.t_tt1 = GLlocal(96,64)

        # self.Graph1=Graph_Attention_Union(64,64)
        # self.Graph2=Graph_Attention_Union(32,32)

        # self.fusion8 = GLlocal(768, 64)
        # self.fusion4 = GLlocal(384, 64)
        self.fusion2 = Fusion(192,32)
        self.fusion1 = Fusion(96,64)



        # self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dn1 = BasicConv2d(128, 96, kernel_size=1)
        self.dn2 = BasicConv2d(256, 192, kernel_size=1)
        # self.avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sm1 = BasicConv2d(96, 32, kernel_size=1)
        self.sm2 = BasicConv2d(192, 64, kernel_size=1)
        # self.sm1 = BasicConv2d(256, 768, kernel_size=3, stride=4, padding=1)
        # self.up22 = BasicConv2d(256, 192, kernel_size=3, stride=1, padding=1)


        # self.selfdc_16 = EDFM(64, 64)
        self.selfdc_8 = EDFM(64, 64)
        self.selfdc_4 = EDFM(64, 64)
        self.selfdc_2 = EDFM(32,32)
        self.selfdc_1 = EDFM(32,32)





        self.fdm = FDM()



    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]
        # in_data_1 =  self.layer0_r (in_data)

        in_data_1 = self.layer0(in_data)
        in_data_1_detail = self.layer0_detail(in_data)
        del in_data
        in_data_1_d = self.layerd0(in_depth)
        in_data_1d_detail = self.layerd0_detail(in_depth)
        del in_depth

        in_data_2 = self.layer1(in_data_1)
        in_data_2_d = self.layerd1(in_data_1_d)
        in_data_2_detail = self.layer1_detail(in_data_1_detail)
        in_data_2d_detail = self.layerd1_detail(in_data_1d_detail)
        in_data_4 = self.layer2(in_data_2)
        in_data_4_d = self.layerd2(in_data_2_d)

        in_data_8 = self.layer3(in_data_4)
        in_data_8_d = self.layerd3(in_data_4_d)


        in_data_8_dd=self.trans8(in_data_8_d)
        in_data_4_dd = self.trans4(in_data_4_d)
        in_data_2_dd = self.trans2(in_data_2_d)
        in_data_1_dd = self.trans1(in_data_1_d)
        # in_data_8=in_data_8+self.sm(in_data_2_detail)
        # in_data_8_d=in_data_8_d+self.sm(in_data_2d_detail)
        # in_data_16 = self.encoder16(in_data_8)
        # in_data_16_d = self.depth_encoder16(in_data_8_d)  +self.up1(in_data_1d_detail)
        #
        se1=self.t_tt1(in_data_1, in_data_1_d)
        se2 = self.t_tt2(in_data_2, in_data_2_d)

        rsd1=self.t_tt1(in_data_1,self.dn1(in_data_1d_detail))
        tsd1 = self.t_tt1(in_data_1_d, self.dn1(in_data_1_detail))

        # in_data_11_detail=self.sm1(self.dn1(in_data_1_detail))
        # in_data_11d_detail = self.sm1(self.dn1(in_data_1d_detail))
#################
        rsd2=self.t_tt2(in_data_2,self.dn2(in_data_2d_detail))
        tsd2 = self.t_tt2(in_data_2_d, self.dn2(in_data_2_detail))
#################
        # in_data_22_detail=self.sm2(self.dn2(in_data_2_detail))
        # in_data_22d_detail = self.sm2(self.dn2(in_data_2d_detail))

        # in_data_s1=self.fusion1(in_data_1,in_data_1_d)
        # in_data_s2 = self.fusion2(in_data_2, in_data_2_d)

        in_data_11=in_data_1+self.dn1(in_data_1_detail)
        in_data_22 =in_data_2+self.dn2(in_data_2_detail)
        in_data_11_d=in_data_1_d+self.dn1(in_data_1d_detail)
        in_data_22_d = in_data_2_d+self.dn2(in_data_2d_detail)
        # in_16=in_data_16+in_data_16_d

        in_data_1_aux = self.t_trans1(in_data_11, in_data_11_d,se1,rsd1,tsd1)
        in_data_2_aux = self.t_trans2(in_data_22, in_data_22_d, se2, rsd2, tsd2)
        # in_data_1_aux=in_data_1_aux*detail1+in_data_1_aux+in_data_1_aux*in_data_s1+in_data_1_aux
        # in_data_2_aux = in_data_2_aux * detail2 + in_data_2_aux + in_data_2_aux * in_data_s2 + in_data_2_aux

        in_data_4_aux = self.t_trans4(in_data_4, in_data_4_d)
        in_data_8_aux = self.t_trans8(in_data_8, in_data_8_d)
        # in_data_16_aux = self.t_trans16(in_data_16, in_data_16_d)

        in_data_1 = self.trans1(in_data_1)
        in_data_2 = self.trans2(in_data_2)
        in_data_4 = self.trans4(in_data_4)
        in_data_8 = self.trans8(in_data_8)
        # in_data_16 = self.trans16(in_data_16)

        out_data_8 = in_data_8
        out_data_8 = self.upconv8(out_data_8)  # 1024

        # out_data_8 = self.upsample_add(self.selfdc_16(out_data_16, in_data_16_aux), in_data_8)
        # out_data_8 = self.upconv8(out_data_8)  # 512

        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4)
        out_data_4 = self.upconv4(out_data_4)  # 256

        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux), in_data_2)
        out_data_2 = self.upconv2(out_data_2)  # 64

        out_data_1 = self.upsample_add(self.selfdc_2(out_data_2, in_data_2_aux), in_data_1)
        out_data_1 = self.upconv1(out_data_1)  # 32

        out_data = self.fdm(out_data_1, out_data_2, out_data_4, out_data_8)

        # in_data_111=self.sm1(in_data_11)
        # in_data_111_d=self.sm1(in_data_11_d)
        # in_data_222 = self.sm2(in_data_22)
        # in_data_222_d = self.sm2(in_data_22_d)

        return out_data, in_data_8,in_data_8_dd, in_data_4,in_data_4_dd,  in_data_2,in_data_2_dd, in_data_1, in_data_1_dd


def fusion_model():
    model = DEFNet()
    return model
def Teacher_model():
    model = Teacher()
    return model

if __name__ == "__main__":
    model = DEFNet()
    x = torch.randn(2,3,256,256)
    depth = torch.randn(2,3,256,256)
    fuse = model([x,depth])
    print(fuse.shape)