import torch
from tqdm import tqdm
from pathlib import Path
from nlut_models import *
import torch.utils.data as data
from PIL import Image
from utils.losses import *
from parameter_finetuning import *
from torch.utils import data
import torch.nn as nn
from torchvision.utils import save_image
import time
import numpy as np
import os, sys
import shutil
import cv2
import imageio
import gc
from fp_model import BiSeNet
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print(f'now device is {device}')

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts in BGR format
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], # background (0), face, left eyebrow
                   [255, 0, 85], [255, 0, 170], # right eyebrow, left eye
                   [0, 255, 0], [85, 255, 0], [170, 255, 0], # right eye, [], left ear
                   [0, 255, 85], [0, 255, 170], # right ear, []
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],  # nose, [], upper lip
                   [0, 85, 255], [0, 170, 255], # lower lip, neck
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],  # [], clothing (16), hair (17)
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],  # hat (18)
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    # Modified olors for all 20 parts in BGR format - only background, clothes, accesories, hair, and rest of face
    modified_part_colors = [[0, 0, 0], [255, 0, 0], [255, 0, 0], # background (0), face, left eyebrow
                   [255, 0, 0], [0, 0, 255], # right eyebrow, left eye
                   [0, 0, 255], [0, 0, 0], [255, 0, 0], # right eye, [], left ear
                   [255, 0, 0], [0, 0, 0], # right ear, []
                   [255, 0, 0], [0, 0, 0], [255, 0, 0],  # nose, [], upper lip
                   [255, 0, 0], [255, 0, 0], # lower lip, neck
                   [0, 0, 0], [0, 255, 255], [0, 255, 0],  # [], clothing (16), hair (17)
                   [255, 0, 255], [0, 0, 0], [0, 0, 0],  # hat (18)
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # for pi in range(0, len(part_colors)):
    #     index = np.where(vis_parsing_anno == pi)
    #     vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    for pi in range(0, len(modified_part_colors)):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = modified_part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0) # Change weighting since we don't care about visualization

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return Image.fromarray(cv2.cvtColor(vis_im, cv2.COLOR_BGR2RGB))

def get_seg(input_frame, model):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = input_frame
        # image = img.resize((512, 512), Image.BILINEAR)
        image = img
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = model(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

    return vis_parsing_maps(image, parsing, stride=1)

# Broken
def get_rvm_seg(input_frame, rvm_net):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Set initial recurrent states to None
    rec = [None] * 4
    with torch.no_grad():
        img = input_frame
        # image = img.resize((512, 512), Image.BILINEAR)
        image = img
        img = to_tensor(image)
        print("Converted to tensor!")
        img = torch.permute(img, (2, 0, 1))
        print("Permuted!")
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        print("Passed to device!")
        print(torch.size(img))
        fgr, pha, *rec = model(img, *rec, downsample_ratio=0.25)
        print("Got pha!")

        seg = (pha > 0.5).float()

        # Convert the segmentation map to a NumPy array
        seg_np = seg.squeeze(0).cpu().numpy()
        print(np.shape(seg_np))
        

        # print(parsing)
        # print(np.unique(parsing))

    return seg_np

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for src_dir, dirs, files in os.walk(src_folder):
        dst_dir = src_dir.replace(src_folder, dst_folder, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            if file_.endswith('.avi') or file_.endswith('.png') or file_.endswith('.mat'):
                continue
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                # os.remove(dst_file)
                continue
            shutil.copy2(src_file, dst_dir)

def read_ubfc_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        print(np.shape(frames))
        return np.asarray(frames)

# Resize back to original size
def resize_to_original(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def face_detection(frame, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.
    Args:
        frame(np.array): a single frame.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    detector = cv2.CascadeClassifier('/playpen-nas-ssd/akshay/UNC_Google_Physio/MA-rPPG-Video-Toolbox/utils/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if len(face_zone) < 1:
        print("ERROR: No Face Detected")
        face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    elif len(face_zone) >= 2:
        face_box_coor = np.argmax(face_zone, axis=0)
        face_box_coor = face_zone[face_box_coor[2]]
        print("Warning: More than one faces are detected(Only cropping the biggest one.)")
    else:
        face_box_coor = face_zone[0]
    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor


def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def adjust_learning_rate(optimizer, iteration_count, opt):
    """Imitating the original implementation"""
    # lr = opt.lr / (1.0 + opt.lr_decay * iteration_count)
    lr = opt.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def p_transform():
    transform_list = [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def train_transform2():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def finetuning_train(opt, original=None, example=None):
    content_tf = train_transform()
    style_tf = train_transform()

    if 1:
        content_images = content_tf(original).unsqueeze(0).to(device)
        content_images = content_images.repeat(opt.batch_size, 1, 1, 1)
    else:
        content_dataset = FlatFolderDataset(opt.content_dir, content_tf)
        content_iter = iter(data.DataLoader(
            content_dataset, batch_size=opt.batch_size,
            sampler=InfiniteSamplerWrapper(content_dataset),
            num_workers=opt.n_threads))
    if 1:
        style_images = style_tf(example).unsqueeze(0).to(device)
        style_images = style_images.repeat(opt.batch_size, 1, 1, 1)
    else:
        style_dataset = FlatFolderDataset(opt.style_dir, style_tf)
        style_iter = iter(data.DataLoader(
            style_dataset, batch_size=opt.batch_size,
            sampler=InfiniteSamplerWrapper(style_dataset),
            num_workers=opt.n_threads))
    if opt.batch_size == 1:
        # content_images = content_images
        # style_images = style_images
        content_images = torch.cat([content_images, content_images], dim=0)
        style_images = torch.cat([style_images, style_images], dim=0)

    model = NLUTNet(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("--------no checkpoint found---------")
    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    # optimizer = torch.optim.Adam(model.module.parameters(), lr=opt.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    log_c = []
    log_s = []
    # log_mse = []
    Time = time.time()

    losses = AverageMeter()
    c_losses = AverageMeter()
    s_losses = AverageMeter()
    # mse_losses = AverageMeter()
    tv_losses = AverageMeter()
    mn_losses = AverageMeter()

    # -----------------------training------------------------
    for i in range(opt.start_iter, opt.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i, opt=opt)
        if original == None:
            content_images = next(content_iter).to(device)
        if example == None:
            style_images = next(style_iter).to(device)

        stylized, st_out, others = model(
            content_images, content_images, style_images, TVMN=TVMN_temp)
        tvmn = others.get("tvmn")
        mn_cons = opt.lambda_smooth * \
            (tvmn[0]+10*tvmn[2]) + opt.lambda_mn*tvmn[1]

        loss_c, loss_s = model.encoder(content_images, style_images, stylized)
        loss_c = loss_c.mean()
        loss_s = loss_s.mean()

        # loss_mse = mseloss(content_images, stylized)
        loss_style = opt.content_weight*loss_c + opt.style_weight * \
            loss_s + opt.mn_cons_weight*mn_cons  # +tv_cons

        # optimizer update
        optimizer.zero_grad()
        loss_style.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()

        # update loss log
        log_c.append(loss_c.item())
        log_s.append(loss_s.item())
        # log_mse.append(loss_mse.item())

        losses.update(loss_style.item())
        c_losses.update(loss_c.item())
        s_losses.update(loss_s.item())
        # mse_losses.update(loss_mse.item())
        mn_losses.update(mn_cons.item())

        # save image
        if i % opt.print_interval == 0 or (i + 1) == opt.max_iter:

            if opt.batch_size == 1:
                content_image, style_image, stylized = content_images[
                    :1], style_images[:1], stylized[:1]
                output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)
                output_images = torch.cat(
                    (content_image.cpu(), style_image.cpu(), stylized.cpu()), 0)
                save_image(stylized.cpu(), output_name, nrow=opt.batch_size)
            else:
                output_name = os.path.join(opt.save_dir, "%06d.jpg" % i)
                output_images = torch.cat(
                    (content_images.cpu(), style_images.cpu(), stylized.cpu()), 0)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print("iter %d   time/iter: %.2f  lr: %.6f loss_mn: %.4f loss_c: %.4f   loss_s: %.4f losses: %.4f " % (i,
                                                                                                                   (time.time(
                                                                                                                   )-Time)/opt.print_interval,
                                                                                                                   current_lr,
                                                                                                                   mn_losses.avg,
                                                                                                                   c_losses.avg, s_losses.avg,
                                                                                                                   losses.avg
                                                                                                                   ))
            log_c = []
            log_s = []
            Time = time.time()

        if (i + 1) % opt.save_model_interval == 0 or (i + 1) == opt.max_iter:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))

            state = {'iter': i, 'state_dict': state_dict,
                     'optimizer': optimizer.state_dict()}
            torch.save(state, opt.resume)
            torch.save(state, "./"+opt.save_dir+"/" +
                       str(i)+"_finetuning_style_lut.pth")

     # Clean-up
    model.cpu()
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def get_lut(opt, original, example):

    # opt = setting.opt
    model = NLUTNet(opt.model, dim=opt.dim).to(device)
    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("--------loading checkpoint----------")
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            # opt.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])

        else:
            print("--------no checkpoint found---------")

    model.train()
    TVMN_temp = TVMN(opt.dim).to(device)

    content_tf2 = train_transform2()
    content_images = content_tf2(original).unsqueeze(0).to(device)
    style_images = content_tf2(example).unsqueeze(0).to(device)

    content_images = content_images.repeat(2, 1, 1, 1)
    style_images = style_images.repeat(2, 1, 1, 1)

    stylized, st_out, others = model(
        content_images, content_images, style_images, TVMN=TVMN_temp)
    # save_image(stylized, "output_name.png", nrow=opt.batch_size)

    LUT = others.get("LUT")

    # Clean-up
    model.cpu()
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    return LUT[:1]


# def draw_video(target_mask, original_path, reference_mask, corrected_mask, LUT):
def draw_video(target_mask, corrected_mask, LUT, face_parse_model):
    sigmod_infer = nn.Sigmoid()
    # cap_target_src = cv2.VideoCapture(target_mask)
    # src_true, original = cap_target_src.read()

    # example = reference_mask

    # 一些变换 toPIL&nb

    # LUT = infer(original_path, example)
    content_tf = p_transform()
    TrilinearInterpo = TrilinearInterpolation()

    # Path(corrected_mask).parent.mkdir(parents=True, exist_ok=True)
    # cap_target = cv2.VideoCapture(target_mask)
    # cap_reference = content_tf(reference_mask).unsqueeze(0).to(device)

    # print(target_mask.shape)
    # width = target_mask.shape[2]
    # height = target_mask.shape[1]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30

    # cap_corrected = cv2.VideoWriter(
        # corrected_mask, fourcc, fps, (width, height))

    # frame_count = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = target_mask.shape[0]

    all_time = 0
    frame = 0
    restyled_video = []

    try:
        with tqdm(desc=f"Frames", total=frame_count) as pbar:
            for i in range(frame_count):
                target = target_mask[i]
                original_image = np.copy(target)
                target_seg = get_seg(target, face_parse_model)
                # cv2.imwrite("Test_target_seg_frame.png", cv2.cvtColor(np.array(target_seg), cv2.COLOR_RGB2BGR))

                # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
                target = Image.fromarray(target)
                target = content_tf(target).unsqueeze(0).to(device)

                start_time = time.time()

                img_res = TrilinearInterpo(LUT, target)
                img_out = img_res+target

                # img_out = sigmod_infer(img_out)

                img_out = torch.squeeze(img_out, dim=0)
                img_out = torch.permute(img_out, (1, 2, 0))

                # 结束时间
                end_time = time.time()
                all_time = all_time+(end_time-start_time)

                corrected = img_out.detach().cpu().numpy()*255
                corrected = np.uint8(np.clip(corrected, 0, 255))

                # cv2.imwrite("Test_corrected_frame.png", cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
                # cv2.imwrite("Test_original_frame.png", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

                target_seg = np.array(target_seg)
                # Get the skin mask from the target segmentation map
                skin_mask = np.all(target_seg == [0, 0, 255], axis=-1)

                # Replace the skin pixels in initial target with the skin pixels from the corrected output
                original_image[np.where(skin_mask)] = corrected[np.where(skin_mask)]

                corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

                # cv2.imwrite("Test_original_frame_after_replace.png", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                # exit()
                restyled_video.append(original_image)
                # cap_corrected.write(corrected)
                pbar.update(1)
                frame = frame + 1
            print(f'all fps: {fps/all_time}')
            average_time = 1000.0*all_time/frame  # ms
            print(f'average time: {average_time}')
    finally:
        print('Saving video to: {}'.format(corrected_mask))
        np.save(corrected_mask, restyled_video)
        # cap_target.release()
        # cap_corrected.release()

# def draw_video(target_mask, corrected_mask, LUT, face_parse_model):
#     sigmod_infer = nn.Sigmoid()
#     # cap_target_src = cv2.VideoCapture(target_mask)
#     # src_true, original = cap_target_src.read()

#     # example = reference_mask

#     # 一些变换 toPIL&nb

#     # LUT = infer(original_path, example)
#     content_tf = p_transform()
#     TrilinearInterpo = TrilinearInterpolation()

#     # Path(corrected_mask).parent.mkdir(parents=True, exist_ok=True)
#     # cap_target = cv2.VideoCapture(target_mask)
#     # cap_reference = content_tf(reference_mask).unsqueeze(0).to(device)

#     # print(target_mask.shape)
#     # width = target_mask.shape[2]
#     # height = target_mask.shape[1]
#     # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = 30

#     # cap_corrected = cv2.VideoWriter(
#         # corrected_mask, fourcc, fps, (width, height))

#     # frame_count = int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_count = target_mask.shape[0]

#     all_time = 0
#     frame = 0
#     restyled_video = []

#     try:
#         with tqdm(desc=f"Frames", total=frame_count) as pbar:
#             initial_target = target_mask[0]
#             initial_target_rvm = np.copy(initial_target)
#             cv2.imwrite("Test_initial_target.png", cv2.cvtColor(initial_target, cv2.COLOR_RGB2BGR))

#             initial_target_seg = get_seg(initial_target, face_parse_model)
#             cv2.imwrite("Test_initial_target_seg_frame.png", cv2.cvtColor(np.array(initial_target_seg), cv2.COLOR_RGB2BGR))

#             # print("I got here!")
#             # initial_target_rvm_seg = get_rvm_seg(initial_target_rvm, rvm_net)

#             # exit()

#             target = Image.fromarray(initial_target)
#             target = content_tf(target).unsqueeze(0).to(device)

#             img_res = TrilinearInterpo(LUT, target)
#             img_out = img_res+target
#             img_out = torch.squeeze(img_out, dim=0)
#             img_out = torch.permute(img_out, (1, 2, 0))

#             corrected = img_out.detach().cpu().numpy()*255
#             corrected = np.uint8(np.clip(corrected, 0, 255))

#             cv2.imwrite("Test_corrected_frame.png", cv2.cvtColor(np.array(corrected), cv2.COLOR_RGB2BGR))

#             initial_target = np.array(initial_target)
#             corrected = np.array(corrected)
#             initial_target_seg = np.array(initial_target_seg)
#             print(initial_target_seg)

#             # # Get the skin mask from the initial segmentation map
#             # skin_mask = (initial_target_seg == [0, 0, 255])
#             skin_mask = np.all(initial_target_seg == [0, 0, 255], axis=-1)

#             print(np.shape(initial_target_seg))
#             print(np.shape(skin_mask))

#             print(initial_target[0,0,:])
#             print(corrected[0,0,:])
#             # Replace the skin pixels in initial target with the skin pixels from the corrected output
#             # cv2.imwrite("Test_for_shading_map_target_2.png", cv2.cvtColor(initial_target, cv2.COLOR_RGB2BGR))
#             new_target = np.copy(initial_target)
#             # cv2.imwrite("Test_for_shading_map_target_2.png", cv2.cvtColor(initial_target, cv2.COLOR_RGB2BGR))
#             new_target[np.where(skin_mask)] = corrected[np.where(skin_mask)]
#             # cv2.imwrite("Test_for_shading_map_target_3.png", cv2.cvtColor(initial_target, cv2.COLOR_RGB2BGR))

#             # save the corrected image
#             # cv2.imwrite("Test_corrected_frame_seg_mod.png", cv2.cvtColor(new_target, cv2.COLOR_RGB2BGR))

#             # cv2.imwrite("Test_for_shading_map_target.png", cv2.cvtColor(initial_target, cv2.COLOR_RGB2BGR))
#             # cv2.imwrite("Test_for_shading_map_corrected.png", cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
#             # Convert images to float
#             initial_target = np.float32(initial_target)
#             corrected = np.float32(corrected)

#             print(initial_target[0,0,:])
#             print(corrected[0,0,:])

#             # Divide original image by restyled image and ignore divide by zero errors
#             # with np.errstate(divide='ignore', invalid='ignore'):
#             shading_map = cv2.divide(corrected, initial_target)

#             # Replace NaNs with 0
#             shading_map[np.isnan(shading_map)] = 0

#             # Replace inf values with the maximum non-inf value in the shading map
#             max_val = np.max(shading_map[np.isfinite(shading_map)])
#             shading_map[np.isinf(shading_map)] = max_val

#             print(shading_map[0,0,:])
#             print(np.max(shading_map))

#             # # Save the shading map for visualization
#             shading_map_norm = cv2.normalize(shading_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#             # cv2.imwrite("shading_map_opencv.png", shading_map_norm)

#             print(np.max(shading_map_norm))
#             print(shading_map_norm[0,0,:])

#             # # Verify that the shading map is correct by trying to reconstruct restyled_img1
#             # reconstructed_restyled_img1 = cv2.multiply(og_img1, shading_map)

#             # # Now, apply the same shading_map to all frames in the original video
#             # restyled_video = []
#             # for frame in original_video:
#             #     frame = np.float32(frame)
#             #     restyled_frame = cv2.multiply(frame, shading_map)
#             #     restyled_video.append(restyled_frame)
#             #     cv2.imwrite("intermediate_restyled_video_frame.png", cv2.cvtColor(restyled_frame, cv2.COLOR_RGB2BGR))

#             # # Save the restyled video as an npy file
#             # restyled_video = np.uint8(restyled_video)

#             for i in range(frame_count):
#                 start_time = time.time()

#                 # Apply shading map to each frame
#                 target_seg = get_seg(target_mask[i], face_parse_model)
#                 target = np.float32(target_mask[i])
#                 cv2.imwrite("Test_loop_target_before_inpaint.png", cv2.cvtColor(np.array(target_mask[i]), cv2.COLOR_RGB2BGR))
#                 restyled_frame = cv2.multiply(target, shading_map)

#                 target = np.uint8(target)
#                 restyled_frame = np.uint8(restyled_frame)
#                 target_seg = np.array(target_seg)
#                 # Get the skin mask from the target segmentation map
#                 skin_mask = np.all(target_seg == [0, 0, 255], axis=-1)

#                 # Replace the skin pixels in initial target with the skin pixels from the corrected output
#                 target[np.where(skin_mask)] = restyled_frame[np.where(skin_mask)]

#                 # 结束时间
#                 end_time = time.time()
#                 all_time = all_time+(end_time-start_time)

#                 cv2.imwrite("Test_loop_restyled_frame.png", cv2.cvtColor(np.array(restyled_frame), cv2.COLOR_RGB2BGR))
#                 cv2.imwrite("Test_loop_target_after_inpaint.png", cv2.cvtColor(np.array(target), cv2.COLOR_RGB2BGR))
#                 restyled_video.append(target)
#                 # cap_corrected.write(corrected)
#                 pbar.update(1)
#                 frame = frame + 1
#             print(f'all fps: {fps/all_time}')
#             average_time = 1000.0*all_time/frame  # ms
#             print(f'average time: {average_time}')
#     finally:
#         print('Saving video to: {}'.format(corrected_mask))
#         np.save(corrected_mask, np.uint8(restyled_video))
#         exit()
#         # cap_target.release()
#         # cap_corrected.release()

def draw_img(original, dst, LUT):
    content_tf2 = p_transform()
    target = content_tf2(original).unsqueeze(0).to(device)

    TrilinearInterpo = TrilinearInterpolation()
    img_res = TrilinearInterpo(LUT, target)
    img_out = img_res+target

    save_image(img_out, dst, nrow=1)

if __name__ == '__main__':

    # python3 inference_finetuning_video.py --pretrained ./experiments/336999_style_lut.pth 
    # --src_video /playpen-nas-hdd/UNC_Google_Physio/UBFC-rPPG/DATASET_2_backup/DATASET_2/train 
    # --style_path /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/FFHQ_eg3d_dark_skin_tones 
    # --dst_video /playpen-nas-ssd/akshay/UNC_Google_Physio/datasets/NLUT_UBFC_DST --max_iter 100
    opt = parser.parse_args()

    copy_folder(opt.src_video, opt.dst_video)

    # Load the segmentation model on second GPU device
    n_classes = 19
    face_parse_model = BiSeNet(n_classes=n_classes)
    face_parse_model.cuda()
    face_parse_model.load_state_dict(torch.load('/playpen-nas-ssd/akshay/UNC_Google_Physio/lighting/face-parsing.PyTorch/79999_iter.pth'))
    face_parse_model.eval()

    # Load the pre-trained model for RVM if used
    # rvm_net = torch.hub.load('PeterL1n/RobustVideoMatting', 'mobilenetv3').eval().cuda()

    source_list = sorted(os.listdir(opt.src_video))
    style_list = sorted(os.listdir(opt.style_path))
    output_list = sorted(os.listdir(opt.dst_video))

    file_num = len(source_list)
    choose_range = range(0, file_num)

    pbar = tqdm(list(choose_range))

    for i in choose_range:
        source_name = os.fsdecode(source_list[i])
        src_video = read_ubfc_video(os.path.join(opt.src_video, source_name, f'{source_name}_vid.avi'))

        cropped_frames = []
        face_region_all = []

        # First, compute the median bounding box across all frames
        for frame in src_video:
            face_box = face_detection(frame, True, 2.0) # MAUBFC and others
            face_region_all.append(face_box)
        face_region_all = np.asarray(face_region_all, dtype='int')
        face_region_median = np.median(face_region_all, axis=0).astype('int')

        # Apply the median bounding box for cropping and subsequent resizing
        for frame in src_video:
            cropped_frame = frame[int(face_region_median[1]):int(face_region_median[1]+face_region_median[3]),
                                int(face_region_median[0]):int(face_region_median[0]+face_region_median[2])]
            resized_frame = resize_to_original(cropped_frame, np.shape(src_video)[2], np.shape(src_video)[1])
            cropped_frames.append(resized_frame)

        # Cropped source video
        src_video = np.asarray(cropped_frames)

        # Get a frame of the source video
        original = Image.fromarray(src_video[0])

        # Get random reference style
        style_path = np.random.choice(style_list, 1)[0]
        style_filename = os.fsdecode(style_path)  
        style_name = os.path.splitext(style_filename)[0]  
        example = Image.open(os.path.join(opt.style_path, style_path)).convert('RGB')

        # Get destination path
        dst_video = os.path.join(opt.dst_video, source_name, f'{source_name}_{style_name}_vid.npy')
        # output_path = os.path.join(opt.dst_video, source_name, f'{source_name}_{style_name}_vid.npy')
        # np.save(output_path, restyled_video)

        finetuning_train(opt, original, example)
        lut = get_lut(opt, original, example)
        draw_video(src_video, dst_video, lut, face_parse_model)

    # content_video = read_ubfc_video(opt.src_video)

    # original = opt.content_path
    # example = opt.style_path
    # src_video = opt.src_video
    # dst_video = opt.dst_video

    # print(original)
    # print(example)
    # print(src_video)
    # print(dst_video)

    # src_video = read_ubfc_video(opt.src_video)

    # cropped_frames = []
    # face_region_all = []

    # # First, compute the median bounding box across all frames
    # for frame in src_video:
    #     face_box = face_detection(frame, True, 2.0) # MAUBFC and others
    #     face_region_all.append(face_box)
    # face_region_all = np.asarray(face_region_all, dtype='int')
    # face_region_median = np.median(face_region_all, axis=0).astype('int')

    # # Apply the median bounding box for cropping and subsequent resizing
    # for frame in src_video:
    #     cropped_frame = frame[int(face_region_median[1]):int(face_region_median[1]+face_region_median[3]),
    #                         int(face_region_median[0]):int(face_region_median[0]+face_region_median[2])]
    #     resized_frame = resize_to_original(cropped_frame, np.shape(src_video)[2], np.shape(src_video)[1])
    #     cropped_frames.append(resized_frame)

    # src_video = np.asarray(cropped_frames)
    # imageio.mimwrite('./my_output/subject1_input.mp4', src_video, fps=30, codec='libx264')

    # original = Image.fromarray(src_video[0])
    # # original = Image.open('./my_content/subject_1_frame.png').convert('RGB')
    # example = Image.open(opt.style_path).convert('RGB')
    # dst_video = opt.dst_video

    # finetuning_train(opt, original, example)
    # lut = get_lut(opt, original, example)
    # # draw_img(original, './my_output/subject1_out.png', lut)
    # draw_video(src_video, dst_video, lut)
    # print('save to: {}'.format(dst_video))
    