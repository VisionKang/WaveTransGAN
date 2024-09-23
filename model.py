import torch
import torch.nn as nn
import numpy as np
from Networks.net import Conv_small_model, Conv_large_model, Fusion_MODEL, TT_MODEL


class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		self.conv_small_model = Conv_small_model(in_channel=1, out_channel=16)  # 32
		self.tt_transformer1 = TT_MODEL(out_channel=32,
                 img_size=224, patch_size=4, embed_dim=96, num_heads=8, window_size=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True, depth=2,
                 downsample=None, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False)
		self.tt_transformer2 = TT_MODEL(out_channel=96,
										img_size=224, patch_size=4, embed_dim=96, num_heads=8, window_size=1,
										mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
										patch_norm=True, depth=2,
										downsample=None, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False)
		self.conv_large_model = Conv_large_model(in_channel=1, out_channel=32)

		reflection_padding = int(np.floor(3 / 2))
		self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
		self.Conv = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1)
		self.BatchNorm2d = torch.nn.BatchNorm2d(32)
		self.PReLU = torch.nn.PReLU()

		self.fusion_model = Fusion_MODEL(in_channel=32, output_channel=1)

	def forward(self, pre_fus, ir, vis):
		conv1_ir = self.conv_small_model(ir)
		tt_tran_ir = self.tt_transformer1(conv1_ir)

		conv1_vis = self.conv_small_model(vis)
		tt_tran_vis = self.tt_transformer1(conv1_vis)

		conv = self.conv_large_model(pre_fus)
		tt_middle_input = torch.cat([conv, tt_tran_ir, tt_tran_vis], 1)
		tt_middle_output = self.tt_transformer2(tt_middle_input)

		tt_middle_output = self.reflection_pad(tt_middle_output)
		tt_middle_output = self.Conv(tt_middle_output)
		tt_middle_output = self.BatchNorm2d(tt_middle_output)
		tt_middle_output = self.PReLU(tt_middle_output)

		output = self.fusion_model(tt_middle_output)

		return output

class Discriminator_ir(nn.Module):
	"""docstring for Discriminator_v"""
	def __init__(self):
		super(Discriminator_ir, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 3, 2),
			nn.LeakyReLU(0.2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2))

		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2))

		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2))

		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Flatten())

		self.fc = nn.Sequential(
			nn.Linear(256 * 6 * 6, 1),
			nn.Tanh())

	def forward(self, v):

		v = self.conv1(v)
		v = self.conv2(v)
		v = self.conv3(v)
		v = self.conv4(v)
		v = self.conv5(v)
		v = self.fc(v)
		v = v / 2 + 0.5

		return v

class Discriminator_vis(nn.Module):
	"""docstring for Discriminator_i"""
	def __init__(self):
		super(Discriminator_vis, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 3, 2),
			nn.LeakyReLU(0.2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2))

		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2))

		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2))

		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Flatten())

		self.fc = nn.Sequential(
			nn.Linear(256 * 6 * 6, 1),
			nn.Tanh())

	def forward(self, i):

		i = self.conv1(i)
		i = self.conv2(i)
		i = self.conv3(i)
		i = self.conv4(i)
		i = self.conv5(i)
		i = self.fc(i)
		i = i / 2 + 0.5

		return i

class DDcGAN(nn.Module):
	"""docstring for DDcGAN"""
	def __init__(self, if_train=False):
		super(DDcGAN, self).__init__()
		self.if_train = if_train

		self.G = Generator()
		self.D_ir = Discriminator_ir()
		self.D_vis = Discriminator_vis()


	def forward(self, pre_fus, ir, vis):
		fusion = self.G(pre_fus, ir, vis)

		if self.if_train:
			score_vis = self.D_vis(vis)
			score_ir = self.D_ir(ir)
			score_g_vis = self.D_vis(fusion)
			score_g_ir = self.D_ir(fusion)
			return score_vis, score_ir, score_g_vis, score_g_ir, fusion
		else:
			return fusion

