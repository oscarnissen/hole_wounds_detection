import math

import numpy as np
import os
from utils import parse_args
import cv2
import sys
from pathlib import Path



# Downsamples image x number (reduce_factor) of times.
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

# python main.py -i "094759 Scw M12 - Moving No 2 To No3.m4v" -o "images" > log2.out
def main(input_directory):



    ############# LOAD FILES FROM DIRECTORY
    vid_writer, vid_writerL, vid_writerR = None, None, None
    p = Path(input_directory)
    save_path = str(p / p.name)  # img.jpg
    print("input_directory: ", input_directory)
    out_file = str(p) + ".m4v"
    out_file_L = str(p) + "_L.m4v"
    out_file_R = str(p) + "_R.m4v"
    print("out_directory: ", out_file, out_file_L, out_file_R)
    fps = 25
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release previous video writer
    if isinstance(vid_writerL, cv2.VideoWriter):
        vid_writerL.release()  # release previous video writer
    if isinstance(vid_writerR, cv2.VideoWriter):
        vid_writerR.release()  # release previous video writer

    images = [x for x in os.listdir(input_directory)]
    print("im.size(): ", images[0])
    f = os.path.join(input_directory, images[0])
    im = cv2.imread(f)
    h, w, c = im.shape
    print("im.size(): ", h, w, c)
    w = int(w / 2)

    vid_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
    vid_writerL = cv2.VideoWriter(out_file_L, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    vid_writerR = cv2.VideoWriter(out_file_R, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i in images:
    #for filename in os.listdir(input_directory):
        f = os.path.join(input_directory, i)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            im = cv2.imread(f)

            #h, w, c = im.shape
            #print("im.size(): ", im.shape)
            #w = int(w / 2)
            imL = im[:, :w]
            imR = im[:, w:]
            print("imL.size(): ", imL.shape)
            print("imR.size(): ", imR.shape)

            vid_writer.write(im)
            vid_writerL.write(imL)
            vid_writerR.write(imR)
    vid_writer.release()
    vid_writerL.release()
    vid_writerR.release()
    print("DONE!")
    '''
    

    print("cv2.CAP_PROP_FRAME_WIDTH: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          " cv2.CAP_PROP_FRAME_HEIGHT: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2 - 1)
    imgL = frame[:, :w + 1]
    imgR = frame[:, w + 1:]
    num = 0
    # while(cap.isOpened() & num < 650):
    while (cap.isOpened()):
        # if num > 650: break
        num += 1
        # cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        ret, frame = cap.read()
        # cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', gray)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2 - 1)
        # print("width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        #      " height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        #      " FPS: ", cap.get(cv2.CAP_PROP_FPS),
        #      " COUNT: ", cap.get(cv2.CAP_PROP_FRAME_COUNT),
        #      " w: ", w,
        #      "num: ", num)

        imgL = frame[:, :w + 1]
        imgR = frame[:, w + 1:]

        input_file_name = os.path.splitext(input_file)[0]
        filenameL = out_directory + "/stereoLeft/" + input_file_name + "L.jpg"
        filenameR = out_directory + "/stereoRight/" + input_file_name + "R.jpg"
        cv2.imwrite(filenameL, imgL)
        cv2.imwrite(filenameR, imgR)
        print("Saving files! ", filenameL, " ", filenameR)

    cv2.destroyAllWindows()
    '''



if __name__ == "__main__":
    args = parse_args()
    # E:/Calibration Videos/20220824_100141 Calibration
    main(args.source)

