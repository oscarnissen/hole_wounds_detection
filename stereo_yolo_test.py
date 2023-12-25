import sys
import math

sys.path.append("stereo_yolo")
sys.path.append("yolov7")
from pathlib import Path
from numpy import random
import time
import os
import torch

import numpy as np
import cv2

from utils import parse_args
from stereo_yolo.stereo_yolo import StereoYolo
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size

## dataset
from yolov7.utils.datasets import LoadImages

from yolov7.utils.general import strip_optimizer, increment_path, set_logging
from yolov7.utils.torch_utils import select_device, TracedModel, load_classifier, time_synchronized
from yolov7.utils.plots import plot_one_box, plot_3d_bbox, downsample_image, plot_3d_bbox_direction, plot_3d_bbox_flow

# 3D object projection
sys.path.append("stereo_camera_manager")  # for distance calculation
from stereo_camera_manager.utils.triangulation import Triangulation

######## DeepSORT -> Importing DeepSORT
sys.path.append("deep_sort")  # for object tracking
from deep_sort_track import TrackDeepSort


# new_track:
# [62, 3,
# 829.323464665456, 1017.3143939431836, 1110.0960980430064, 1142.3399733423594,
# 155.6561452116596,
# 0.9059469103813171,
# -14.950356975110935, -0.07209658399756336, -15.950515193916544,
# -0.1771060788196337, -1.395586932177082, 375.3881611794781]
def draw_3d_depth_and_speed(im0L, im0R, pred_lst, names, colors, save_img):
    class_colors = colors[0]
    line = []
    for p_el in pred_lst:
        print("p_el: (", p_el[0], p_el[1], p_el[2], p_el[3], p_el[4], p_el[5], ")")
        if p_el[13] > 0:
            spd = f'{p_el[13] / 100:.2f}'
            speed_lbl = f'speed {p_el[13] / 100:.2f} m/s'
        else:
            spd = f' '
            speed_lbl = f' '
        cls = f'fish'  # int(p_el[0])

        if float(p_el[0]) > -1:
            track_id = f'{p_el[0]}'
            color = class_colors[p_el[0] % len(colors)]
        else:
            track_id = f' '
            color = class_colors[0]

        label = f'{cls} {track_id} - {p_el[7] * 100:.2f}%'
        cls = 0

        # cls, track_id, speed, distance, x, y, h, w, dx, dy, dh, dw, dz
        line.append(cls)
        line.append(track_id)
        line.append(spd)
        line.append(p_el[6])  # distance
        line.append(p_el[2])  # x, y, h, w
        line.append(p_el[3])
        line.append(p_el[4])
        line.append(p_el[5])
        line.append(p_el[8])  # dx, dy, dh, dw, dz
        line.append(p_el[9])
        line.append(p_el[10])
        line.append(p_el[11])
        line.append(p_el[12])
        # [x, y, h, w] p_el[2], p_el[3], p_el[4], p_el[5]
        # dxyxy = [dx, dy, dh, dw, dz]
        #dxyxy = [p_el[8], p_el[9], p_el[10], p_el[11], p_el[12]]
        #wh = [p_el[4], p_el[5]]
        ####
        # cls = f'fish'  # int(p_el[0])
        # label = f'{cls} {track_id} - {p_el[7] * 100:.2f}%'
        # track_id = f'{p_el[0]}'  # float(p_el[0]) > -1
        # speed_lbl = f'speed {p_el[13] / 100:.2f} m/s'  # p_el[13] > 0:
        # depth_lbl = f'distance {p_el[6] / 100:.2f} m'  # (p_el[6] > 0 or p_el[6] < 100000)
        ####

        if (p_el[6] > 0 or p_el[6] < 100000):
            depth_lbl = f'distance {p_el[6] / 100:.2f} m'
            ## 3D object projection
            tri = Triangulation()
            xyxyL, xyxyR = tri.find_3dBBox([p_el[2], p_el[3], p_el[4], p_el[5]], p_el[6], im0L, im0R)
            print("xyxyL: ", xyxyL)
            print("xyxyR: ", xyxyR)
            xyxy = [xyxyL, xyxyR]
            # dxyxy = [dx, dy, dh, dw, dz]
            dxyxy = [p_el[8], p_el[9], p_el[10], p_el[11], p_el[12]]
            wh = [p_el[4], p_el[5]]
            xyxyL_prev = []
            xyxyR_prev = []

            for el in p_el[14]:
                L_prev, R_prev = tri.find_3dBBox([el[0], el[1], el[2], el[3]], el[4], im0L, im0R)
                xyxyL_prev.append(L_prev)
                xyxyR_prev.append(R_prev)

            if save_img:  # Add bbox to image
                # plot_3d_bbox(xyxy, im0L, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                #             line_thickness=1)
                #fcls = int(p_el[0] % len(colors[0]))
                #flow_color = colors[0][fcls]
                print("colors: ", colors)
                print("color:", color)
                print("flow_color: p_el[0] ", p_el[0], " - len(colors) ", len(colors), " - p_el[0] % len(colors)", p_el[0] % len(colors))
                #print("flow_color: ", flow_color)

                plot_3d_bbox_flow(xyxy, dxyxy, wh, xyxyL_prev, im0L, #flow_color=flow_color,
                                  side="left", label=label, depth=depth_lbl,
                                  speed=speed_lbl,
                                  color=color, line_thickness=1)
                # plot_3d_bbox(xyxy, im0R, label=label, depth=depth_lbl, speed=speed_lbl, color=colors[cls],
                #             line_thickness=1)
                plot_3d_bbox_flow(xyxy, dxyxy, wh, xyxyR_prev, im0R, #flow_color=flow_color,
                                  side="right", label=label, depth=depth_lbl,
                                  speed=speed_lbl,
                                  color=color, line_thickness=1)
        else:
            depth_lbl = f' '
        print(depth_lbl)

    return line


def calculate_speed(frame_idx, track_fish_list, track_fish_speed):
    track_fish_list_new = []
    print("frame_idx: ", frame_idx)
    print("track_fish_list: ", track_fish_list)
    print("track_fish_speed: ", track_fish_speed)
    for track in track_fish_list:
        # find the track id that corresponds to the track_id
        print("track: ", track)
        #  [62,
        #  844.2738216405669,
        #  1017.3864905271812,
        #  1126.046613236923,
        #  1142.517079421179,
        #  157.0517321438367,
        #  0.9104070663452148]
        [id, x2, y2, w2, h2, z2, c2] = track
        print("id ", id, " x2 ", x2, " y2 ", y2, " w2 ", w2, " h2 ", h2, " z2 ", z2, " c2 ", c2)
        obj = None
        if len(track_fish_speed) > 0:
            print("len(track_fish_speed) ", len(track_fish_speed))
            obj = next((obj for obj in track_fish_speed if obj[0] == id), None)

        if not (obj is None):
            print("obj: ", obj)
            [_, f, x1, y1, w1, h1, z1, _, _, _, _, _, _, _, p1] = obj
            # if id == track_id:
            # if any match
            # calculate the displacement and speed
            dx = x2 - x1
            dy = y2 - y1
            dw = w2 - w1
            dh = h2 - h1
            dz = z2 - z1
            speed = (math.sqrt(dx * dx + dy * dy + dz * dz)) * (25 / (frame_idx - f))
            p2 = [x1, y1, w1, h1, z1]
            if p1 is None:
                p1 = p2
            else:
                p1.append(p2)
            new_track = [id, frame_idx, x2, y2, w2, h2, z2, c2, dx, dy, dw, dh, dz, speed, p1]

        else:
            new_track = [id, frame_idx, x2, y2, w2, h2, z2, c2, 0, 0, 0, 0, 0, 0, []]
        if not new_track == []:
            print("new_track: ", new_track)
            track_fish_list_new.append(new_track)

    print("track_fish_list_new: ", track_fish_list_new)
    return track_fish_list_new


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
    (save_dir / 'img' if save_img else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


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
    vid_path, vid_writer, vid_writerL = None, None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride, stereo_mode=stereomod, stereo_map=stereomap)
    # Get names and colors
    names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
    colors = [[random.randint(0, 255) for _ in range(20)] for _ in names]
    print("colors: ", colors)

    # Run inference
    if device.type != 'cpu':
        yolo_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolo_model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    model = StereoYolo(yolo_model).to(device)
    t0 = time.time()
    ###### DEEPSORT ###########
    dt, seen = [0.0, 0.0, 0.0], 0
    frame_idx = 0
    track_fish = TrackDeepSort(deepsort_model)
    ###########################

    frame = getattr(dataset, 'frame', 0)
    track_fish_speed = []
    for path, imgL, imgR, im0sL, im0sR, vid_cap in dataset:

        t1 = time_synchronized()
        p = path
        im0L = im0sL
        im0R = im0sR

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        print("p.name: ", p.name, " p.name.split:", str(p.name).split())
        file_n, file_ext = str(p.name).split(".", 2)
        save_pathL = str(save_dir / file_n) + "_L." + str(file_ext)
        #txt_path = str(save_dir / 'labels' / p.stem) + (
        #    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        txt_path = str(save_dir / p.name) # + (
            #    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt

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
        features = model([imgL, imgR], [im0sL, im0sR], args)
        t4 = time_synchronized()
        dt[2] += t4

        print("##### stereo_yolo_test, after running the model ##### ")
        print("len(features): ", len(features), "features: ", features)
        frame_idx = frame_idx + 1

        bboxes = [bbox[3] for bbox in features]
        scores = [score[4] for score in features]
        for i, det in enumerate(features):  # detections per image
            seen += 1
            det.append(-1)

        print("bboxes: ", bboxes)
        print("scores: ", scores)
        # DeepSORT -> Saving Track predictions into a text file.
        # save_format = '{frame},{id},{x1},{y1},{w},{h},{x},{y},{z}\n'
        # cls, track_id, speed, distance, x, y, h, w, dx, dy, dh, dw, dz
        save_format = '{frame},{cls},{track_id},{speed},{distance},{x},{y},{h},{w},{dx},{dy},{dh},{dw},{dz} '
        print("#### call to track with temp ################")
        track_fish_list = track_fish.stereo_track_fish(im0L, bboxes, scores)
        t5 = time_synchronized()
        track_fish_speed = calculate_speed(frame_idx, track_fish_list, track_fish_speed)
        print("track_fish_speed: ", track_fish_speed)
        print("features: ", features)
        line = draw_3d_depth_and_speed(im0L, im0R, track_fish_speed, names, colors, (save_img or view_img))
        line.insert(0, frame_idx)

        if save_txt:  # Write to file
            with open(txt_path + '.txt', 'a') as f:
                print("line: ", line)
                #f.write(str(line))
                for l in line:
                    f.write(str(l) + '\t')
                f.write('\n')

        # Stream results
        if view_img:
            cv2.imshow(str(p), downsample_image(
                np.concatenate((
                    im0sL, im0sR
                ), axis=1), 3))
            cv2.waitKey(1)  # 1 millisecond
        # Print time (inference + NMS)
        print(
            f'Done. ({(1E3 * (t3 - t1)):.1f}ms) Inference, ({(1E3 * (t4 - t3)):.1f}ms) NMS, ({(1E3 * (t5 - t4)):.1f}ms) tracking, {(1E3 * (t5 - t1)):.1f}ms) overall')

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
                    if isinstance(vid_writerL, cv2.VideoWriter):
                        vid_writerL.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("fps: ", fps, " w: ", w, " h: ", h)
                    else:  # stream
                        fps, w, h = 30, im0L.shape[1], im0L.shape[0]
                        save_path += '.mp4'
                        save_pathL += '_L.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writerL = cv2.VideoWriter(save_pathL, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(w/2), h))

                print("save_path: ", save_path)
                print("save_pathL: ", save_pathL)
                im0 = im0L.copy()
                vid_writer.write(np.concatenate((
                    im0L, im0R
                ), axis=1))
                #vid_writer.write(im0)
                vid_writerL.write(im0)
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                ####################################################
                ## for debug remove later
                #print("save_path", {save_path})
                # basename_without_ext = os.path.splitext(os.path.basename(save_path))[0]
                filename = str(save_dir / 'img' / file_n)  # img.jpg #os.path.dirname(save_path)
                print("filename: ", {filename})
                cv2.imwrite(filename + "_" + str(frame_idx).zfill(6) + ".jpg", np.concatenate((
                    im0, im0R
                ), axis=1))
                print(f" The image with the result is saved in: {filename}_" + str(frame_idx).zfill(6) + ".jpg")
                ####################################################


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    args = parse_args()

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['yolov7.pt']:
                run_detect(args)
                strip_optimizer(args.weights)
        else:
            run_detect(args)
