import os
import cv2 as cv
from ultralytics import YOLO

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim  = (width, height)

    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    return resized


def filter_tracks(centers, patience):
    filter_dict = {}
    for k, i in centers.items():
        d_frames = i.items()
        filter_dict[k] = dict(list(d_frames)[-patience:])

    return filter_dict


def update_tracking(centers_old, obj_center, thr_centers, lastKey, frame, frame_max):
    is_new = 0
    lastpos = [(k, list(center.keys())[-1], list(center.values())[-1]) for k, center in centers_old.items()]
    lastpos = [(i[0], i[2]) for i in lastpos if abs(i[1] - frame) <= frame_max]

    previous_pos = [(k, obj_center) for k, centers in lastpos if (np.linalg.norm(np.array(centers) - np.array(obj_center)) < thr_centers)]

    if previous_pos:
        id_obj = previous_pos[0][0]
        centers_old[id_obj][frame] = obj_center
    else:
        if lastKey:
            last = lastKey.split('ID')[1]
            id_obj = 'ID' + str(int(last) + 1)
        else:
            id_obj = 'ID0'

        is_new = 1
        centers_old[id_obj] = { frame:obj_center }
        lastKey = list(centers_old.keys())[-1]

    return centers_old, id_obj, is_new, lastKey


verbose = False
scale_percent = 100
conf_level = 0.8
thr_centers = 20
frame_max = 5
patience = 100
alpha = 0.1
class_IDS = [0]
centers_old = {}
obj_id = 0
end = []
frames_list = []
count_p = 0
lastKey = ''


if __name__ == '__main__':
    # https://www.kaggle.com/code/paulojunqueira/yolo-v8-people-detection-and-tracking-in-roi/output?select=yolov8x.pt
    # YOLOv8 engine was inherited from human detection and tracking in ROI from KAGGLE
    model = YOLO('yolov8x.pt')
    dict_classes = model.model.names

    video_files = [f for f in os.listdir('./sample') if f.lower().endswith(('.mp4'))]

    for video_file in video_files:
        print("-"*20, video_file, "-"*20)
        video = cv.VideoCapture('./sample/' + video_file)

        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        fps = video.get(cv.CAP_PROP_FPS)

        if scale_percent != 100:
            print('[INFO] - Scaling change may cause errors in pixels lines ')
            width = int(width * scale_percent / 100)
            height = int(height * scale_percent / 100)
            print('[INFO] - Dim Scaled: ', (width, height))

        #-------------------------------------------------------
        ### Video output ####
        video_name = video_file
        output_path = "./result/" + video_name
        VIDEO_CODEC = "mp4v"

        output_video = cv.VideoWriter(output_path,
                                       cv.VideoWriter_fourcc(*VIDEO_CODEC),
                                       fps, (width, height))

        for i in tqdm(range(int(video.get(cv.CAP_PROP_FRAME_COUNT)))):
            _, frame = video.read()

            frame = resize_frame(frame, scale_percent)

            area_roi = [np.array([(0, 0), (height, 0), (height, width) ,(0, width)], np.int32)]
            # area_roi = [np.array([(1250, 400), (750, 400), (700, 800), (1200, 800)], np.int32)]
            ROI = frame[:, :]

            if verbose:
                print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

            y_hat = model.predict(ROI, conf=conf_level, classes=class_IDS, device=0, verbose=False)

            boxes = y_hat[0].boxes.xyxy.cpu().numpy()
            conf = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy()
            positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.boxes,
                                           columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
            labels = [dict_classes[i] for i in classes]

            # For each people, draw the bounding-box and counting each one the pass thought the ROI area
            for ix, row in enumerate(positions_frame.iterrows()):
                # Getting the coordinates of each vehicle (row)
                xmin, ymin, xmax, ymax, confidence, category, = row[1].astype('int')

                # Calculating the center of the bounding-box
                center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

                # Updating the tracking for each object
                centers_old, id_obj, is_new, lastKey = update_tracking(centers_old, (center_x, center_y), thr_centers,
                                                                       lastKey, i, frame_max)

                # Updating people in roi
                count_p += is_new

                # drawing center and bounding-box in the given frame
                cv.rectangle(ROI, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # box
                for center_x, center_y in centers_old[id_obj].values():
                    cv.circle(ROI, (center_x, center_y), 5, (0, 0, 255), -1)  # center of box

                # Drawing above the bounding-box the name of class recognized.
                cv.putText(img=ROI, text=id_obj + ':' + str(np.round(conf[ix], 2)),
                            org=(xmin, ymin - 10), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(0, 0, 255),
                            thickness=1)

            # drawing the number of people
            cv.putText(img=frame, text=f'Counts People in ROI: {count_p}',
                        org=(30, 40), fontFace=cv.FONT_HERSHEY_TRIPLEX,
                        fontScale=1.5, color=(255, 0, 0), thickness=1)

            # Filtering tracks history
            centers_old = filter_tracks(centers_old, patience)
            if verbose:
                print("contador_in", "contador_out")

            # Drawing the ROI area
            overlay = frame.copy()

            cv.polylines(overlay, pts=area_roi, isClosed=True, color=(255, 0, 0), thickness=2)
            cv.fillPoly(overlay, area_roi, (255, 0, 0))
            frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Saving frames in a list
            frames_list.append(frame)
            # saving transformed frames in a output video formaat
            output_video.write(frame)

            print("++++++++\t", i, "th frame of ", video_name, "\t\t+++++++++++")

        # Releasing the video
        output_video.release()
