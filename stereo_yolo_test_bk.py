import sys

sys.path.append("stereo_yolo")
sys.path.append("yolov7")
from pathlib import Path
from numpy import random
import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.backends.cudnn as cudnn
import numpy as np
import cv2

from utils import parse_args
from stereo_yolo.stereo_yolo import StereoYolo
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size

## dataset
from yolov7.utils.datasets import LoadStreams, LoadImages

from yolov7.utils.general import non_max_suppression, strip_optimizer, increment_path, set_logging
from yolov7.utils.torch_utils import select_device, TracedModel, load_classifier, time_synchronized
from yolov7.utils.plots import plot_one_box, plot_3d_bbox, downsample_image


# 3D object projection
sys.path.append("stereo_camera_manager")  # for distance calculation
from stereo_camera_manager.utils.triangulation import Triangulation

######## DeepSORT -> Importing DeepSORT
sys.path.append("deep_sort")  # for object tracking
from deep_sort_track import TrackDeepSort


def merge_lists(pred_lst, track_fish_list):
    # pred_lst_new = []
    print("pred_lst: ", pred_lst)
    print("track_fish_list", track_fish_list)
    for p, t in [(p, t) for p in sorted(pred_lst) for t in sorted(track_fish_list)]:
        print("p: ", p)
        print("t: ", t)
        print("p[1][0] ", p[1][0], " t[1] ", t[1])
        print("p[1][1] ", p[1][1], " t[2] ", t[2])

        # if (int(p[3]) == int(t[1])) and (int(p[4]) == int(t[2])) and \
        #        (int(t[3]) == int(p[5])) and (int(t[4]) == int(p[6])):
        if (abs(p[1][0] - t[1]) < 20) and (abs(p[1][1] - t[2]) < 20):
            print("tracking fish #", t[0])
            p[7] = t[0]
            # p.append(t[0])
        print("updated p: ", p)


# def draw_depth_and_speed(im0sL, im0sR, left_features, right_features, names, colors, save_img):
# [cls, xyxy, conf, box, conf.item(), depth]
#  [
#  tensor(0.),
#  [tensor(1668.), tensor(810.), tensor(1858.), tensor(937.)],
#  tensor(0.71009),
#  [1668.0, 810.0, 190.0, 127.0],
#  0.7100895047187805,
#  tensor(0.76169),
#  tensor(490.62262)
#  ],
def draw_depth_and_speed(im0L, im0R, predL_lst, predR_lst, names, colors, save_img):
    for p_el_R, p_el_L in zip(predR_lst, predL_lst):
        if (p_el_L[6] > 0 or p_el_L[6] < 10000):
            depth_lbl = f'distance {p_el_L[6] / 100:.2f} m'
        else:
            depth_lbl = f' '

        speed_lbl = f'speed {1:.2f} m/s'
        cls = int(p_el_L[0])
        if float(p_el_L[7]) > -1:
            track_id = p_el_L[7]
        else:
            track_id = f' '
        label = f'{names[cls]} {track_id} - {p_el_L[5] * 100:.2f}%'  # id {p_el_L[7]}
        print(depth_lbl)
        if save_img:  # Add bbox to image
            xyxyL = p_el_L[1]
            plot_one_box(xyxyL, im0L, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                         line_thickness=1)
            xyxyR = p_el_R[1]
            plot_one_box(xyxyR, im0R, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                         line_thickness=1)

def draw_3d_depth_and_speed(im0L, im0R, pred_lst, names, colors, save_img):
    for p_el in pred_lst:
        print("p_el: ", p_el)
        speed_lbl = f'speed {1:.2f} m/s'
        cls = int(p_el[0])

        if float(p_el[7]) > -1:
            track_id = p_el[7]
        else:
            track_id = f' '

        label = f'{names[cls]} {track_id} - {p_el[5] * 100:.2f}%'  # id {p_el_L[7]}
        #label = f'{names[cls]} - {p_el[5] * 100:.2f}%'  # id {p_el_L[7]}

        if (p_el[6] > 0 or p_el[6] < 10000):
            depth_lbl = f'distance {p_el[6] / 100:.2f} m'
            ## 3D object projection
            tri = Triangulation()
            #depth = tri.find_depth(center_point_right, center_point_left, im0sR, im0sL)
            #point = (p_el[1][0], p_el[1][1])
            xyxyL, xyxyR = tri.find_3dBBox(p_el[1], p_el[6], im0L, im0R)
            #xyxyL = [x_L, y_L, p_el[3][2], p_el[3][3]]
            #xyxyR = [x_R, y_R, p_el[3][2], p_el[3][3]]
            print("xyxyL: ", xyxyL)
            print("xyxyR: ", xyxyR)

            xyxy = [xyxyL, xyxyR]

            if save_img:  # Add bbox to image
                #xyxyL = p_el[1]
                plot_3d_bbox(xyxy, im0L, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                             line_thickness=1)
                #xyxyR = p_el[1]
                plot_3d_bbox(xyxy, im0R, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                             line_thickness=1)
        else:
            depth_lbl = f' '
        print(depth_lbl)


def run_detect(args):
    source, weights, view_img, save_txt, imgsz, stereomod, \
    stereomap, trace, deepsort_model = args.source, args.weights, args.view_img, \
                                       args.save_txt, args.img_size, args.stereo_mode, \
                                       args.stereo_map, not args.no_trace, args.deepsort_model

    save_img = not args.nosave and not source.endswith('.txt')  # save inference images

    print("Stereo mode: ", stereomod, " save image: ", save_img, " view image: ", view_img)
    print("Stereo map: ", stereomap)
    print("deepSORT model: ", deepsort_model)
    # Directories
    save_dir = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    yolo_model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(yolo_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if trace:
        yolo_model = TracedModel(yolo_model, device, args.img_size)
    if half:
        yolo_model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader,
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride, stereo_mode=stereomod, stereo_map=stereomap)
    # Get names and colors
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print("stereomod: ", stereomod, "stereomap: ", stereomap, "imgsz: ", imgsz, "stride: ", stride)

    # Run inference
    if device.type != 'cpu':
        yolo_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolo_model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    model = StereoYolo(yolo_model).to(device)

    t0 = time.time()

    '''
    ####################################################
    
    imgL_dir = "C:/Users/ayas/OneDrive - SINTEF/Projects/FishMachineInteraction/src/Fish-detection/dataset/fish" \
               "/left_image/"
    imgR_dir = "C:/Users/ayas/OneDrive - SINTEF/Projects/FishMachineInteraction/src/Fish-detection/dataset/fish" \
               "/right_image/"
    dataset_L = LoadImages(imgL_dir, img_size=imgsz, stride=stride, stereo_mode=False,
                           stereo_map="stereo_calibration_map.xml")
    dataset_R = LoadImages(imgR_dir, img_size=imgsz, stride=stride, stereo_mode=False,
                           stereo_map="stereo_calibration_map.xml")
    img_L = []
    img_R = []
    im0s_L = []
    im0s_R = []
    p = ""
    frame = ""
    for path, imgL, imgR, im0sL, im0sR, vid_cap in dataset_L:
        img_L.append(imgL)
        im0s_L.append(im0sL)
        p = path
        frame = getattr(dataset_L, 'frame', 0)
    for path, imgL, imgR, im0sL, im0sR, vid_cap in dataset_R:
        img_R.append(imgL)
        im0s_R.append(im0sL) '''

    '''
    imgL = img_L[0]
    imgR = img_R[0]
    im0sL = im0s_L[0]
    im0L = im0sL
    im0sR = im0s_R[0]
    im0R = im0sR
    print("imgL_dir: ", imgL_dir)
    do_detect = 1
    print("do detect:", do_detect)

    if do_detect > 0:
        ################################### '''

    ###### DEEPSORT ###########
    dt, seen = [0.0, 0.0, 0.0], 0
    frame_idx = 0
    track_fish = TrackDeepSort(deepsort_model)

    ###########################

    frame = getattr(dataset, 'frame', 0)
    for path, imgL, imgR, im0sL, im0sR, vid_cap in dataset:
        t1 = time_synchronized()
        p = path
        im0L = im0sL
        im0R = im0sR

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + (
            '' if dataset.mode == 'image' else f'_{frame}')  # img.txt

        imgL, imgR = torch.from_numpy(imgL).to(device), torch.from_numpy(imgR).to(device)
        imgL = imgL.half() if half else imgL.float()  # uint8 to fp16/32
        imgR = imgR.half() if half else imgR.float()  # uint8 to fp16/32
        imgL /= 255.0  # 0 - 255 to 0.0 - 1.0
        imgR /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgL.ndimension() == 3:
            imgL = imgL.unsqueeze(0)
            # print("imgR.size: ", imgR.size)
        if imgR.ndimension() == 3:
            imgR = imgR.unsqueeze(0)

        t2 = time_synchronized()
        dt[0] += t2 - t1
        # Warmup
        if device.type != 'cpu' and (
                old_img_b != imgL.shape[0] or old_img_h != imgL.shape[2] or old_img_w != imgL.shape[3]):
            print("Working on the GPU .. ")
            old_img_b = imgL.shape[0]
            old_img_h = imgL.shape[2]
            old_img_w = imgL.shape[3]
            for i in range(3):
                yolo_model(imgL, augment=args.augment)[0]
        # Inference
        t3 = time_synchronized()
        dt[1] += t3 - t2

        left_features, right_features, features = model([imgL, imgR], [im0sL, im0sR], args)

        t4 = time_synchronized() - t3
        dt[2] += t4

        print("len(left_features): ", len(left_features), "left_features: ", left_features)
        print("len(right_features): ", len(right_features), "right_features: ", right_features)
        print("len(features): ", len(features), "features: ", features)
        frame_idx = frame_idx + 1


        #bboxes = [bbox[3] for bbox in left_features]
        bboxes = [bbox[3] for bbox in features]
        scores = [score[4] for score in features]
        for i, det in enumerate(features):  # detections per image
            seen += 1
            det.append(-1)

        print("bboxes: ", bboxes)
        print("scores: ", scores)
        # DeepSORT -> Saving Track predictions into a text file.
        save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
        print("bboxesL ", bboxes)
        print("scoresL ", scores)
        track_fish_list = track_fish.track_fish(im0L, bboxes, scores)
        #merge_lists(left_features, track_fish_list)
        merge_lists(features, track_fish_list)
        #draw_depth_and_speed(im0L, im0R, left_features, right_features, names, colors, (save_img or view_img))
        draw_3d_depth_and_speed(im0L, im0R, features, names, colors, (save_img or view_img))
        # Stream results
        if view_img:
            cv2.imshow(str(p), downsample_image(
                np.concatenate((
                    im0sL, im0sR
                ), axis=1), 3))
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, np.concatenate((
                    im0L, im0R
                ), axis=1))
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                print('video or stream')
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("fps: ", fps, " w: ", w, " h: ", h)
                    else:  # stream
                        fps, w, h = 30, im0L.shape[1], im0L.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                print("vid_writer.write(im0L)")
                vid_writer.write(np.concatenate((
                    im0L, im0R
                ), axis=1))
                ####################################################
                ## for debug remove later
                print("save_path", {save_path})
                # basename_without_ext = os.path.splitext(os.path.basename(save_path))[0]
                filename = os.path.dirname(save_path)
                cv2.imwrite(filename + "img.jpg", np.concatenate((
                    im0L, im0R
                ), axis=1))
                print(f" The image with the result is saved in: {filename}img")
                ####################################################


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    args = parse_args()
    #   "--weights ./model/yolo7.pt"

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['yolov7.pt']:
                run_detect(args)
                strip_optimizer(args.weights)
        else:
            run_detect(args)

