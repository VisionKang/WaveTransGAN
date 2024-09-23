import os
import time
from tqdm import trange
import random
import torch
from torch.optim import Adam
from tools.utils import *
from tools.args_fusion import args
import loss
from Networks.model import DDcGAN
from datetime import datetime
import pywt
import pywt.data

def main():
    torch.cuda.empty_cache()
    start_time = datetime.now()

    image_path = list_images(args.images_ir)
    random.shuffle(image_path)

    batch_size = args.batch_size

    SwinFuse_model = DDcGAN(if_train=True)
    if args.cuda:
        SwinFuse_model.cuda()
    print(SwinFuse_model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_con_ssim = loss.final_ssim
    loss_con_sal = loss.L_con_sal
    loss_con_grad = loss.L_con_grad
    loss_con = loss.L_con
    loss_g_adv = loss.L_adv_G
    loss_g = loss.L_G
    loss_d_ir = loss.L_D_ir
    loss_d_vis = loss.L_D_vis

    tbar = trange(args.epochs)
    print('Start training.....')

    all_Loss_con_ssim_list = []
    all_Loss_con_sal_list = []
    all_Loss_con_grad_list = []
    all_Loss_con_list = []
    all_Loss_g_adv_list = []
    all_Loss_g_list = []
    all_Loss_d_ir_list = []
    all_Loss_d_vis_list = []

    count_loss = 0
    all_Loss_con_ssim = 0.
    all_Loss_con_sal = 0.
    all_Loss_con_grad = 0.
    all_Loss_con = 0.
    all_Loss_g_adv = 0.
    all_Loss_g = 0.
    all_Loss_d_ir = 0.
    all_Loss_d_vis = 0.
    epoch_num = 0
    for e in tbar:
        epoch_num = epoch_num + 1
        print('Epoch %d.....' % e)

        random.shuffle(image_path)

        images_ir_list = []
        images_vis_list = []
        images_ir_sal_list = []
        images_vis_sal_list = []

        for i in image_path:
            original_images_ir_path = os.path.join(args.images_ir, i)
            original_images_vis_path = os.path.join(args.images_vis, i)
            original_images_ir_sal_path = os.path.join(args.images_ir_sal, i)
            original_images_vis_sal_path = os.path.join(args.images_vis_sal, i)
            images_ir_list.append(original_images_ir_path)
            images_vis_list.append(original_images_vis_path)
            images_ir_sal_list.append(original_images_ir_sal_path)
            images_vis_sal_list.append(original_images_vis_sal_path)

        original_images_ir = get_train_images_auto(images_ir_list)
        original_images_vis = get_train_images_auto(images_vis_list)
        original_images_ir_sal = get_train_images_auto(images_ir_sal_list)
        original_images_vis_sal = get_train_images_auto(images_vis_sal_list)

        original_images_ir, batches = load_dataset(original_images_ir, batch_size)
        original_images_vis, batches = load_dataset(original_images_vis, batch_size)
        original_images_ir_sal, batches = load_dataset(original_images_ir_sal, batch_size)
        original_images_vis_sal, batches = load_dataset(original_images_vis_sal, batch_size)

        SwinFuse_model.train()
        count = 0
        print('begin batch training')

        for batch in range(batches):

            original_images_ir_input = original_images_ir[batch * batch_size:(batch * batch_size + batch_size)]
            original_images_vis_input = original_images_vis[batch * batch_size:(batch * batch_size + batch_size)]
            original_images_ir_sal_input = original_images_ir_sal[batch * batch_size:(batch * batch_size + batch_size)]
            original_images_vis_sal_input = original_images_vis_sal[batch * batch_size:(batch * batch_size + batch_size)]

            coeffs21 = pywt.dwt2(original_images_ir_input, 'haar')
            coeffs22 = pywt.dwt2(original_images_vis_input, 'haar')

            LL1, (LH1, HL1, HH1) = coeffs21
            LL2, (LH2, HL2, HH2) = coeffs22

            LL = 0.5 * LL1 + 0.5 * LL2
            LH = 0.5 * LH1 + 0.5 * LH2
            HL = 0.5 * HL1 + 0.5 * HL2
            HH = 0.5 * HH1 + 0.5 * HH2

            coeffs2 = LL, (LH, HL, HH)
            pre_fusion_image = pywt.idwt2(coeffs2, 'haar')

            original_images_ir_input = torch.from_numpy(original_images_ir_input).float()
            original_images_ir_input = (original_images_ir_input - 127.5) / 127.5
            original_images_vis_input = torch.from_numpy(original_images_vis_input).float()
            original_images_vis_input = (original_images_vis_input - 127.5) / 127.5
            pre_fusion_image = torch.from_numpy(pre_fusion_image).float()
            pre_fusion_image = (pre_fusion_image - 127.5) / 127.5
            original_images_ir_sal_input = torch.from_numpy(original_images_ir_sal_input).float()
            original_images_ir_sal_input = (original_images_ir_sal_input - 127.5) / 127.5
            original_images_vis_sal_input = torch.from_numpy(original_images_vis_sal_input).float()
            original_images_vis_sal_input = (original_images_vis_sal_input - 127.5) / 127.5

            count += 1

            if args.cuda:
                original_images_ir_input = original_images_ir_input.cuda()
                original_images_vis_input = original_images_vis_input.cuda()
                sal_ir = original_images_ir_sal_input.cuda()
                sal_vis = original_images_vis_sal_input.cuda()
                pre_fusion_image = pre_fusion_image.cuda()

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = True
            optimizer_vis = Adam(SwinFuse_model.parameters(), args.lr)
            for _ in range(2):
                score_vis, _, score_g_vis, _, _ = SwinFuse_model(pre_fusion_image, original_images_ir_input, original_images_vis_input)
                Loss_d_vis = loss_d_vis(score_vis, score_g_vis)
                optimizer_vis.zero_grad()
                Loss_d_vis.backward()
                optimizer_vis.step()

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = True
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = False
            optimizer_ir = Adam(SwinFuse_model.parameters(), args.lr)
            for _ in range(2):
                _, score_ir, _, score_g_ir, _ = SwinFuse_model(pre_fusion_image, original_images_ir_input, original_images_vis_input)
                Loss_d_ir = loss_d_ir(score_ir, score_g_ir)
                optimizer_ir.zero_grad()
                Loss_d_ir.backward()
                optimizer_ir.step()

            for i in SwinFuse_model.G.parameters():
                i.requires_grad = True
            for i in SwinFuse_model.D_ir.parameters():
                i.requires_grad = False
            for i in SwinFuse_model.D_vis.parameters():
                i.requires_grad = False
            optimizer_g = Adam(SwinFuse_model.parameters(), args.lr)
            score_vis, score_ir, score_g_vis, score_g_ir, fusion = SwinFuse_model(pre_fusion_image, original_images_ir_input, original_images_vis_input)
            Loss_g = loss_g(original_images_ir_input, original_images_vis_input, fusion, sal_ir, sal_vis, device,
                            args.alpha,
                            args.beta, args.gamma, score_g_ir, score_g_vis, args.lamda)

            optimizer_g.zero_grad()
            Loss_g.backward()
            optimizer_g.step()

            Loss_con_ssim = loss_con_ssim(original_images_ir_input, original_images_vis_input, fusion)
            Loss_con_sal = loss_con_sal(original_images_ir_input, original_images_vis_input, sal_ir, sal_vis, fusion)
            Loss_con_grad = loss_con_grad(original_images_ir_input, original_images_vis_input, fusion, device)
            Loss_con = loss_con(original_images_ir_input, original_images_vis_input, fusion, sal_ir, sal_vis, device, args.alpha,
                                args.beta, args.gamma)
            Loss_g_adv = loss_g_adv(score_g_ir, score_g_vis)
            Loss_g = loss_g(original_images_ir_input, original_images_vis_input, fusion, sal_ir, sal_vis, device, args.alpha,
                            args.beta, args.gamma, score_g_ir, score_g_vis, args.lamda)
            Loss_d_ir = loss_d_ir(score_ir, score_g_ir)
            Loss_d_vis = loss_d_vis(score_vis, score_g_vis)

            all_Loss_con_ssim += Loss_con_ssim.item()
            all_Loss_con_sal += Loss_con_sal.item()
            all_Loss_con_grad += Loss_con_grad.item()
            all_Loss_con += Loss_con.item()
            all_Loss_g_adv += Loss_g_adv.item()
            all_Loss_g += Loss_g.item()
            all_Loss_d_ir += Loss_d_ir.item()
            all_Loss_d_vis += Loss_d_vis.item()

            if (batch + 1) % args.log_interval == 0:
                elapsed_time = datetime.now() - start_time
                print("lr: %s, elapsed_time: %s\n" % (args.lr, elapsed_time))
                mesg = "{}\tEpoch {}:\t[{}/{}]\t Loss_con_ssim: {:.6f}\t Loss_con_int: {:.6f}\t " \
                       "Loss_con_grad: {:.6f}\t Loss_con: {:.6f}\t Loss_g_adv: {:.6f}\t all_Loss_g: {:.6f}\t " \
                       "all_Loss_d_ir: {:.6f}\t all_Loss_d_vis: {:.6f}".format(
                                  time.ctime(), e + 1, count, batches,
                                  all_Loss_con_ssim / args.log_interval,
                                  all_Loss_con_sal / args.log_interval,
                                  all_Loss_con_grad / args.log_interval,
                                  all_Loss_con / args.log_interval,
                                  all_Loss_g_adv / args.log_interval,
                                  all_Loss_g / args.log_interval,
                                  all_Loss_d_ir / args.log_interval,
                                  all_Loss_d_vis / args.log_interval)
                tbar.set_description(mesg)
                all_Loss_con_ssim_list.append(all_Loss_con_ssim / args.log_interval)
                all_Loss_con_sal_list.append(all_Loss_con_sal / args.log_interval)
                all_Loss_con_grad_list.append(all_Loss_con_grad / args.log_interval)
                all_Loss_con_list.append(all_Loss_con / args.log_interval)
                all_Loss_g_adv_list.append(all_Loss_g_adv / args.log_interval)
                all_Loss_g_list.append(all_Loss_g / args.log_interval)
                all_Loss_d_ir_list.append(all_Loss_d_ir / args.log_interval)
                all_Loss_d_vis_list.append(all_Loss_d_vis / args.log_interval)

                count_loss = count_loss + 1

                all_Loss_con_ssim = 0.
                all_Loss_con_sal = 0.
                all_Loss_con_grad = 0.
                all_Loss_con = 0.
                all_Loss_g_adv = 0.
                all_Loss_g = 0.
                all_Loss_d_ir = 0.
                all_Loss_d_vis = 0.

            if (batch + 1) % args.log_save_model_interval == 0:
                save_model_filename = "epoch" + str(epoch_num) + "_" + "batch" + str(batch+1) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(SwinFuse_model.state_dict(), save_model_path)
                print("\nDone, trained model saved at", save_model_path)

        save_model_filename = "epoch" + str(epoch_num) + "_" + \
                              str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(SwinFuse_model.state_dict(), save_model_path)
        print("\nDone, trained model saved at", save_model_path)

    SwinFuse_model.eval()
    SwinFuse_model.cpu()


if __name__ == "__main__":
    main()
