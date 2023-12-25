import itertools

import os
from utils import parse_args

import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import sys

sys.path.append("yolov7")  # for object detection
sys.path.append("stereo_camera_manager")  # for distance calculation
sys.path.append("deep_sort")  # for object tracking

######## Distance calculation including stereo camera calibration
from stereo_camera_manager.utils.triangulation import Triangulation
from stereo_camera_manager.utils.calibration import Calibration

######## Object detection with Yolov7
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box, downsample_image
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

######## DeepSORT -> Importing DeepSORT
from deep_sort_track import TrackDeepSort


def merge_lists(pred_lst, track_fish_list):
    #pred_lst_new = []
    print("pred_lst: ", pred_lst)
    print("track_fish_list", track_fish_list)
    for p, t in [(p, t) for p in sorted(pred_lst) for t in sorted(track_fish_list)]:
        print("p: ", p)
        print("t: ", t)
        print("p[1][0] ", p[1][0], " t[1] ", t[1])
        print("p[1][1] ", p[1][1], " t[2] ", t[2])

        #if (int(p[3]) == int(t[1])) and (int(p[4]) == int(t[2])) and \
        #        (int(t[3]) == int(p[5])) and (int(t[4]) == int(p[6])):
        if (abs(p[1][0]-t[1]) < 20) and (abs(p[1][1]-t[2]) < 20):
            print("tracking fish #", t[0])
            p[7] = t[0]
            #pred_lst_new.append([p, t[0]])
        #else:
        #    pred_lst_new.append([p[0], p[1], p[2], '-1'])
    #return pred_lst_new


def match_stereo_detection(predL_l, predR_l):
    predL_lst = []
    predR_lst = []

    for l, r in [(l, r) for l in sorted(predL_l) for r in sorted(predR_l)]:
        print(l, r)
        if (abs(l[1][0] - r[1][0]) < 50) and (abs(l[1][1] - r[1][1]) < 50) and \
                (abs(l[1][2] - r[1][2]) < 50) and (abs(l[1][3] - r[1][3]) < 50):
            conf = (l[2] + r[2]) / 2
            l.append(conf)
            r.append(conf)
            predL_lst.append(l)
            predR_lst.append(r)

    return predL_lst, predR_lst


def compute_depth_and_speed(im0L, im0R, im0sL, im0sR, predL_lst, predR_lst, names, colors, save_img):
    predL_lst, predR_lst = match_stereo_detection(predL_lst, predR_lst)
    # Stereo vision setup parameters
    print("Stereo mode - Calculating the depth ##### ")
    print("Found ", len(predL_lst), " fish!")
    if len(predR_lst) == len(predL_lst):
        for p_el_R, p_el_L in zip(predR_lst, predL_lst):
            center_point_right = (p_el_R[1][0], p_el_R[1][1])
            center_point_left = (p_el_L[1][0], p_el_L[1][1])
            tri = Triangulation()
            depth = tri.find_depth(center_point_right, center_point_left, im0sR, im0sL)
            depth_lbl = f'distance {depth / 100:.2f} m'
            speed_lbl = f'speed {1:.2f} m/s'

            print(depth_lbl)
            if save_img:  # Add bbox to image
                x_L = p_el_L[1][0]
                y_L = p_el_L[1][1]
                xyxyL = p_el_L[1]
                clsL = int(p_el_L[0])
                print("xyxyL: ", xyxyL, "x_L: ", x_L, "y_L: ", y_L, f' conf: {p_el_R[2] * 100:.2f}%')
                labelL = f'{names[clsL]} {p_el_L[3]} - {p_el_L[2] * 100:.2f}%'
                # labelL = f'{names[clsL]}  {p_el_L[3] * 100:.2f}%'
                plot_one_box(xyxyL, im0L, label=labelL, depth=depth_lbl, speed=speed_lbl, color=colors[clsL],
                             line_thickness=1)

                x_R = p_el_R[1][0]
                y_R = p_el_R[1][1]
                xyxyR = p_el_R[1]
                clsR = int(p_el_R[0])
                #p_el_R[3] = p_el_L[3]
                print("xyxyR: ", xyxyR, "x_R: ", x_R, "y_R: ", y_R, f' conf: {p_el_R[2] * 100:.2f}%')
                labelR = f'{names[clsR]} {p_el_R[3]} - {p_el_R[2] * 100:.2f}%'
                # labelR = f'{names[clsR]} {p_el_R[3] * 100:.2f}%'
                plot_one_box(xyxyR, im0R, label=labelR, depth=depth_lbl, speed=speed_lbl, color=colors[clsR],
                             line_thickness=1)


#predR_lst.append([clsR, xyxyR, confR,
#                  [bbox_left, bbox_top, bbox_w, bbox_h], confR.item])
def compute_depth(im0sL, im0sR, predL_lst, predR_lst):
    print("##### Before calling match_stereo_detection #########################")
    print("predL_lst", predL_lst)
    print("predR_lst", predR_lst)
    predL_lst, predR_lst = match_stereo_detection(predL_lst, predR_lst)
    print("##### After calling match_stereo_detection #########################")
    print("predL_lst", predL_lst)
    print("predR_lst", predR_lst)

    # Stereo vision setup parameters
    print("Stereo mode - Calculating the depth ##### ")
    print("Found ", len(predL_lst), " fish!")

    for p_el_R, p_el_L in zip(predR_lst, predL_lst):
        print("p_el_L", p_el_L)
        print("p_el_R", p_el_R)
        center_point_right = (p_el_R[1][0], p_el_R[1][1])
        center_point_left = (p_el_L[1][0], p_el_L[1][1])
        tri = Triangulation()
        depth = tri.find_depth(center_point_right, center_point_left, im0sR, im0sL)
        depth_lbl = f'distance {depth / 100:.2f} m'
        p_el_L.append(depth)
        p_el_L.append("-1")
        p_el_R.append(depth)
        p_el_R.append("-1")
        print(predL_lst)
        print(predR_lst)
    print("### AFTER FOR LOOP IN COMPUTE_DEPTH ####")
    print("predL_lst", predL_lst)
    print("predR_lst", predR_lst)
    return predL_lst, predR_lst


# predR_lst.append([clsR, xyxyR, confR,
#     [bbox_left, bbox_top, bbox_w, bbox_h], confR.item])
def draw_depth_and_speed(im0L, im0R, predL_lst, predR_lst, names, colors, save_img):
    for p_el_R, p_el_L in zip(predR_lst, predL_lst):
        depth_lbl = f'distance {p_el_L[6] / 100:.2f} m'
        speed_lbl = f'speed {1:.2f} m/s'

        print(depth_lbl)
        if save_img:  # Add bbox to image
            x_L = p_el_L[1][0]
            y_L = p_el_L[1][1]
            xyxyL = p_el_L[1]
            clsL = int(p_el_L[0])
            print("xyxyL: ", xyxyL, "x_L: ", x_L, "y_L: ", y_L, f' conf: {p_el_R[2] * 100:.2f}%')
            labelL = f'{names[clsL]} {p_el_L[7]} - {p_el_L[2] * 100:.2f}%'
            # labelL = f'{names[clsL]}  {p_el_L[3] * 100:.2f}%'
            plot_one_box(xyxyL, im0L, label=labelL, depth=depth_lbl, speed=speed_lbl, color=colors[clsL],
                         line_thickness=1)

            x_R = p_el_R[1][0]
            y_R = p_el_R[1][1]
            xyxyR = p_el_R[1]
            clsR = int(p_el_R[0])
            p_el_R[7] = p_el_L[7]
            print("xyxyR: ", xyxyR, "x_R: ", x_R, "y_R: ", y_R, f' conf: {p_el_R[2] * 100:.2f}%')
            labelR = f'{names[clsR]} {p_el_R[7]} - {p_el_R[2] * 100:.2f}%'
            # labelR = f'{names[clsR]} {p_el_R[3] * 100:.2f}%'
            plot_one_box(xyxyR, im0R, label=labelR, depth=depth_lbl, speed=speed_lbl, color=colors[clsR],
                         line_thickness=1)


class Yolo7Detect:
    def __init__(self, args):
        self.opt = args

    def run(self):
        with torch.no_grad():
            if self.opt.update:  # update all models (to fix SourceChangeWarning)
                for self.opt.weights in ['yolov7.pt']:
                    self.detect()
                    strip_optimizer(self.opt.weights)
            else:
                self.detect()

    def detect(self, save_img=False):
        source, weights, view_img, save_txt, imgsz, stereomod, \
        stereomap, trace, deepsort_model = self.opt.source, self.opt.weights, self.opt.view_img, \
                                           self.opt.save_txt, self.opt.img_size, self.opt.stereo_mode, \
                                           self.opt.stereo_map, not self.opt.no_trace, self.opt.deepsort_model

        save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        print("Stereo mode: ", stereomod, " save image: ", save_img, " view image: ", view_img)
        print("Stereo map: ", stereomap)
        print("deepSORT model: ", deepsort_model)

        ####### DeepSORT ############################################
        track_fish_L = TrackDeepSort(deepsort_model)
        track_fish_R = TrackDeepSort(deepsort_model)
        ############################################################

        # Directories
        save_dir = Path(
            increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, self.opt.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader,
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, stereo_mode=stereomod, stereo_map=stereomap)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        print("stereomod: ", stereomod, "stereomap: ", stereomap, "imgsz: ", imgsz, "stride: ", stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        ######
        # Run inference.
        # dt, seen = [0.0, 0.0, 0.0], 0
        # frame_idx = 0
        ###### DEEPSORT ###########
        dtL, seenL = [0.0, 0.0, 0.0], 0
        dtR, seenR = [0.0, 0.0, 0.0], 0
        frame_idx = 0
        ###########################

        # for path, img, im0s, vid_cap in dataset:
        for path, imgL, imgR, im0sL, im0sR, vid_cap in dataset:
            t1 = time_synchronized()
            imgL = torch.from_numpy(imgL).to(device)
            imgL = imgL.half() if half else imgL.float()  # uint8 to fp16/32
            imgL /= 255.0  # 0 - 255 to 0.0 - 1.0
            if imgL.ndimension() == 3:
                imgL = imgL.unsqueeze(0)
            # print("img.size: ", imgL.size)
            if stereomod:
                # print("imgR.size: ", imgR.size)
                imgR = torch.from_numpy(imgR).to(device)
                imgR = imgR.half() if half else imgR.float()  # uint8 to fp16/32
                imgR /= 255.0  # 0 - 255 to 0.0 - 1.0
                if imgR.ndimension() == 3:
                    imgR = imgR.unsqueeze(0)
            t2 = time_synchronized()
            dtL[0] += t2 - t1

            # Warmup
            if device.type != 'cpu' and (
                    old_img_b != imgL.shape[0] or old_img_h != imgL.shape[2] or old_img_w != imgL.shape[3]):
                print("Working on the GPU .. ")
                old_img_b = imgL.shape[0]
                old_img_h = imgL.shape[2]
                old_img_w = imgL.shape[3]
                for i in range(3):
                    model(imgL, augment=self.opt.augment)[0]

            # Inference
            predL = model(imgL, augment=self.opt.augment)[0]
            if stereomod:
                predR = model(imgR, augment=self.opt.augment)[0]
            t3 = time_synchronized()
            dtL[1] += t3 - t2

            # Apply NMS
            predL = non_max_suppression(predL, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                        agnostic=self.opt.agnostic_nms)
            if stereomod:
                predR = non_max_suppression(predR, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                            agnostic=self.opt.agnostic_nms)
            t4 = time_synchronized() - t3
            dtL[2] += t4

            # Apply Classifier
            if classify:
                predL = apply_classifier(predL, modelc, imgL, im0sL)
                if stereomod:
                    predR = apply_classifier(predR, modelc, imgR, im0sR)
            frame_idx = frame_idx + 1

            # Process detections
            predL_lst = []
            predR_lst = []
            for i, det in enumerate(predL):  # detections per image
                seenL += 1
                if webcam:  # batch_size >= 1
                    p, sL, im0L, frame = path[i], '%g: ' % i, im0sL[i].copy(), dataset.count
                else:
                    # p, s, im0, frame = path, '', im0sL, getattr(dataset, 'frame', 0)
                    p, sL, sR, im0L, frame = path, '', '', im0sL, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                print("save_path: ", save_path, "txt_path: ", txt_path)
                gn = torch.tensor(im0L.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(imgL.shape[2:], det[:, :4], im0L.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        sL += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                    bboxesL = []
                    scoresL = []
                    for *xyxy, conf, cls in reversed(det):
                        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
                        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
                        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
                        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
                        box = [bbox_left, bbox_top, bbox_w, bbox_h]
                        bboxesL.append(box)
                        scoresL.append(conf.item())

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, bbox_left, bbox_top, bbox_w, bbox_h, conf) if self.opt.save_conf else (cls, *xywh)  # label format

                            with open(txt_path + 'L.txt', 'a') as f:
                                # print(('%g ' * len(line)).rstrip() % line)
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if not stereomod and (save_img or view_img):  # Add bbox to image
                            x_L = int(xyxy[0])
                            y_L = int(xyxy[1])
                            print("xyxyL: ", xyxy, "x_L: ", x_L, "y_L: ", y_L)
                            label = f'{names[int(cls)]} {conf:.2f} \n x_L = {x_L} \n y_L = {y_L}'
                            plot_one_box(xyxy, im0L, label=label, color=colors[int(cls)], line_thickness=1)

                        predL_lst.append([cls, xyxy, conf,
                                          [bbox_left, bbox_top, bbox_w, bbox_h],
                                          conf.item()])


                # Print time (inference + NMS)
                print(f'{sL}Done. ({(1E3 * (t3 - t1)):.1f}ms) Inference, ({(1E3 * (t4 - t3)):.1f}ms) NMS')
                ###########################################
                # right image prediction
                for i, detR in enumerate(predR):  # detections per image
                    seenR += 1
                    # p, s, im0, frame = path, '', im0sL, getattr(dataset, 'frame', 0)
                    sR, im0R, frame = '', im0sR, getattr(dataset, 'frame', 0)

                    gnR = torch.tensor(im0R.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(detR):
                        # Rescale boxes from img_size to im0 size
                        detR[:, :4] = scale_coords(imgR.shape[2:], detR[:, :4], im0R.shape).round()
                        # Print results
                        for c in detR[:, -1].unique():
                            n = (detR[:, -1] == c).sum()  # detections per class
                            sR += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        # DeepSORT -> Extracting Bounding boxes and its confidence scores.
                        bboxesR = []
                        scoresR = []
                        for *xyxyR, confR, clsR in reversed(detR):
                            bbox_left = min([xyxyR[0].item(), xyxyR[2].item()])
                            bbox_top = min([xyxyR[1].item(), xyxyR[3].item()])
                            bbox_w = abs(xyxyR[0].item() - xyxyR[2].item())
                            bbox_h = abs(xyxyR[1].item() - xyxyR[3].item())
                            box = [bbox_left, bbox_top, bbox_w, bbox_h]
                            bboxesR.append(box)
                            scoresR.append(confR.item())

                            if save_txt:  # Write to file
                                xywhR = (xyxy2xywh(torch.tensor(xyxyR).view(1, 4)) / gnR).view(
                                    -1).tolist()  # normalized xywh
                                line = (clsR, *xywhR, bbox_left, bbox_top, bbox_w, bbox_h, confR) if self.opt.save_conf else (clsR, *xywhR)  # label format
                                with open(txt_path + 'R.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            predR_lst.append([clsR, xyxyR, confR,
                                              [bbox_left, bbox_top, bbox_w, bbox_h],
                                              confR.item()])
                ####################################################

                print(f'{sR}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                ################################################################
                ### Calculating the depth ######################################
                if stereomod:
                    predL_lst, predR_lst = compute_depth(im0sL, im0sR, predL_lst, predR_lst)
                    print("------------------")
                    print(predL_lst)
                    print(predR_lst)

                    bboxes = [bbox[3] for bbox in predL_lst]
                    scores = [score[4] for score in predL_lst]
                    print("bboxes: ", bboxes)
                    print("scores: ", scores)

                    # DeepSORT -> Saving Track predictions into a text file.
                    save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
                    print("bboxesL ", bboxesL)
                    print("scoresL ", scoresL)
                    track_fish_list_L = track_fish_L.track_fish(im0L, bboxes, scores)
                    for tid in track_fish_list_L:
                        with open(txt_path + 'L_T.txt', 'a') as f:
                            line = save_format.format(frame=frame_idx, id=tid[0], x1=tid[1],
                                                        y1=tid[2],
                                                        w=tid[3], h=tid[4], x=-1, y=-1,
                                                        z=-1)
                            f.write(line)
                    merge_lists(predL_lst, track_fish_list_L)

                    draw_depth_and_speed(im0L, im0R, predL_lst, predR_lst, names, colors, (save_img or view_img))

                    # compute_depth_and_speed(im0L, im0R, im0sL, im0sR, predL_lst, predR_lst, names, colors,
                    #                        (save_img or view_img))

                ################################################################

                # Stream results
                if view_img:
                    if stereomod:
                        cv2.imshow(str(p), downsample_image(
                            np.concatenate((
                                im0L, im0R
                            ), axis=1), 3))
                    else:
                        cv2.imshow(str(p), im0L)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        if stereomod:
                            cv2.imwrite(save_path, np.concatenate((
                                im0L, im0R
                            ), axis=1))
                        else:
                            cv2.imwrite(save_path, im0L)
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

        print("save_txt: ", save_txt, " save_img: ", save_img)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    args = parse_args()

    print(args)

    ydetect = Yolo7Detect(args)
    ydetect.run()
