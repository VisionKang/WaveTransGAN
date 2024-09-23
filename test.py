import torch
from torch.autograd import Variable
from tools.utils import *
from tools.args_fusion import args
import numpy as np
import time
from Networks.model import DDcGAN
import pywt
import pywt.data

def load_model(path):

    SwinFuse_model = DDcGAN(if_train=False)
    SwinFuse_model.load_state_dict(torch.load(path), True)

    para = sum([np.prod(list(p.size())) for p in SwinFuse_model.parameters()])

    type_size = 4
    print('Model {} : params: {:4f}M'.format(SwinFuse_model._get_name(), para * type_size / 1000 / 1000))

    SwinFuse_model.eval()
    SwinFuse_model.cuda()

    return SwinFuse_model


def run_demo(model, ir_path, vis_path, output_path_root, index):

    img_ir = get_test_images(ir_path)
    img_vis = get_test_images(vis_path)

    coeffs21 = pywt.dwt2(img_ir, 'haar')
    coeffs22 = pywt.dwt2(img_vis, 'haar')

    LL1, (LH1, HL1, HH1) = coeffs21
    LL2, (LH2, HL2, HH2) = coeffs22

    LL = 0.5 * LL1 + 0.5 * LL2
    LH = 0.5 * LH1 + 0.5 * LH2
    HL = 0.5 * HL1 + 0.5 * HL2
    HH = 0.5 * HH1 + 0.5 * HH2

    coeffs2 = LL, (LH, HL, HH)
    img_prefus = pywt.idwt2(coeffs2, 'haar')

    img_ir = torch.from_numpy(img_ir).float()
    img_ir = (img_ir - 127.5) / 127.5
    img_vis = torch.from_numpy(img_vis).float()
    img_vis = (img_vis - 127.5) / 127.5
    img_prefus = torch.from_numpy(img_prefus).float()
    img_prefus = (img_prefus - 127.5) / 127.5

    if args.cuda:
        img_ir = img_ir.cuda()
        img_vis = img_vis.cuda()
        img_prefus = img_prefus.cuda()
    img_ir = Variable(img_ir, requires_grad=False)
    img_vis = Variable(img_vis, requires_grad=False)
    img_prefus = Variable(img_prefus, requires_grad=False)

    img_fusion = model(img_prefus, img_ir, img_vis)
    img_fusion = ((img_fusion / 2) + 0.5) * 255

    file_name = str(index) + '.jpg'
    output_path = output_path_root + file_name
    print(output_path)
    save_image_test(img_fusion, output_path)

def main():
    Output_path = args.output_path
    Ir_path = args.ir_path
    Vis_path = args.vis_path
    model_path = args.model_path_gray

    with torch.no_grad():
        model = load_model(model_path)
        begin = time.time()
        for i in range(4):
            index = i + 1
            ir_path = Ir_path + str(index) + '.jpg'
            vis_path = Vis_path + str(index) + '.jpg'

            run_demo(model, ir_path, vis_path, Output_path, index)
        end = time.time()
        print("consumption time of generating:%s " % (end - begin))
    print('Done......')

if __name__ == '__main__':
    main()
