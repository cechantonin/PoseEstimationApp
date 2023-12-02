import warnings
import numpy as np
import mmcv
import globals

from mmpose.apis import (get_track_id,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_tracking_result,
                         inference_bottom_up_pose_model)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector
from mmtrack.apis import inference_mot
from mmtrack.apis import init_model as init_tracking_model

def get_dist(pose, pose_last):
    dist = 0
    for i in range(17):
        dist += np.sqrt((pose[i, 1] - pose_last[i, 1]) ** 2 + (pose[i, 2] - pose_last[i, 2]) ** 2)
    return dist


def get_iou(bbox, bbox_last):
    x1 = max(bbox[0], bbox_last[0])
    y1 = max(bbox[1], bbox_last[1])
    x2 = min(bbox[2], bbox_last[2])
    y2 = min(bbox[3], bbox_last[3])

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    area_last = (bbox_last[2] - bbox_last[0]) * (bbox_last[3] - bbox_last[1])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = float(area + area_last - intersection)

    if union == 0:
        union = 1e-5
        warnings.warn('Union is 0, setting union to 1e-5.')

    iou = intersection / union

    return iou


def track_iou(results, results_last, next_id, threshold, min_keypoints=3):
    if results_last is None:
        results_last = []

    for result in results:
        if len(results_last) == 0:
            track_id = -1
        else:
            max_iou = -1
            max_index = -1

            for index, res_last in enumerate(results_last):
                iou_score = get_iou(result['bbox'], res_last['bbox'])
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_index = index

            if max_iou > threshold:
                track_id = results_last[max_index]['track_id']
                del results_last[max_index]
            else:
                track_id = -1

        if track_id == -1:
            if np.count_nonzero(result['keypoints'][:, 1]) >= min_keypoints:
                result['track_id'] = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result['keypoints'][:, 1] = -10
                result['bbox'] *= 0
                result['track_id'] = -1
        else:
            result['track_id'] = track_id

    return results, next_id


def track_dist(results, results_last, next_id, threshold, min_keypoints=3):
    if results_last is None:
        results_last = []

    for result in results:
        if len(results_last) == 0:
            track_id = -1
        else:
            min_dist = np.inf
            min_index = -1

            # CALCULATE MINIMUM DISTANCE BETWEEN POSES
            for index, res_last in enumerate(results_last):
                dist = get_dist(result['keypoints'], res_last['keypoints'])
                if dist < min_dist:
                    min_dist = dist
                    min_index = index

            if min_dist < threshold:
                track_id = results_last[min_index]['track_id']
                del results_last[min_index]
            else:
                track_id = -1

        if track_id == -1:
            if np.count_nonzero(result['keypoints'][:, 1]) >= min_keypoints:
                result['track_id'] = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                result['keypoints'][:, 1] = -10
                result['bbox'] *= 0
                result['track_id'] = -1
        else:
            result['track_id'] = track_id

    return results, next_id


def process_mmtracking_results(mmtracking_results):
    person_results = []
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    return person_results


def mmpose_inference(progress_callback, in_file, track='iou', pose='hrnet', bbox_thr=0.3, tracking_thr=0.3,
                     kpt_radius=4, line_thickness=1):
    # Detection
    det_config = './configs/faster_rcnn.py'
    det_checkpoint = './checkpoints/faster_rcnn.pth'

    # Tracking "Greedily by IoU", "Greedily by distance", "Greedily by OKS", "ByteTrack", "OC-SORT", "QDTrack"
    tracking_type = ''
    if track == "Greedily by IoU":
        tracking_type = 'iou'
    elif track == "Greedily by distance":
        tracking_type = 'dist'
    elif track == "Greedily by OKS":
        tracking_type = 'oks'
    elif track == "ByteTrack":
        tracking_type = 'bytetrack'
        track_config = './configs/bytetrack.py'
        track_checkpoint = './checkpoints/bytetrack.pth'
    elif track == "OC-SORT":
        tracking_type = 'ocsort'
        track_config = './configs/ocsort.py'
        track_checkpoint = './checkpoints/ocsort.pth'
    elif track == "QDTrack":
        tracking_type = 'qdtrack'
        track_config = './configs/qdtrack.py'
        track_checkpoint = './checkpoints/qdtrack.pth'

    # Pose "Hourglass", "HRNet", "DEKR", "HigherHRNet"
    if pose == "Hourglass":
        pose_config = './configs/hourglass.py'
        pose_checkpoint = './checkpoints/hourglass.pth'
    elif pose == 'HRNet':
        pose_config = './configs/hrnet.py'
        pose_checkpoint = './checkpoints/hrnet.pth'
    elif pose == "DEKR":
        pose_config = './configs/dekr.py'
        pose_checkpoint = './checkpoints/dekr.pth'
    elif pose == "HigherHRNet":
        pose_config = './configs/higher_hrnet.py'
        pose_checkpoint = './checkpoints/higher_hrnet.pth'

    print('Initializing model...')
    if tracking_type != 'bytetrack' and tracking_type != 'ocsort' and tracking_type != 'qdtrack':
        det_model = init_detector(
            det_config, det_checkpoint, device='cuda:0')
    elif tracking_type != "Tracking disabled":
        tracking_model = init_tracking_model(
            track_config, track_checkpoint, device='cuda:0')

    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device='cuda:0')

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is not None:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(in_file)
    assert video.opened, f'Failed to load video file {in_file}'

    smoother = Smoother(filter_cfg='configs/_base_/filters/one_euro.py', keypoint_dim=2)

    next_id = 1
    pose_results = []
    pose_json = []
    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_results_last = pose_results

        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        if tracking_type != 'bytetrack' and tracking_type != 'ocsort' and tracking_type != 'qdtrack':
            mmdet_results = inference_detector(det_model, cur_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, 1)
        else:
            mmtracking_results = inference_mot(
                tracking_model, cur_frame, frame_id=frame_id)

            # keep the person class bounding boxes.
            person_results = process_mmtracking_results(mmtracking_results)

        # test a single image, with a list of bboxes.
        if pose == 'DEKR' or pose == 'HigherHRNet':
            pose_results, _ = inference_bottom_up_pose_model(
                pose_model,
                cur_frame,
                dataset=dataset,
                dataset_info=dataset_info,
                pose_nms_thr=0.9,
                return_heatmap=False,
                outputs=None)
        else:
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                cur_frame,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)

        if tracking_type != 'bytetrack' and tracking_type != 'ocsort' and tracking_type != 'qdtrack':
            if tracking_type == 'oks':
                pose_results, next_id = get_track_id(
                    pose_results,
                    pose_results_last,
                    next_id,
                    use_oks=True,
                    tracking_thr=tracking_thr)
            elif tracking_type == 'dist':
                pose_results, next_id = track_dist(pose_results, pose_results_last, next_id,
                                                   threshold=tracking_thr * 100)
            elif tracking_type == 'iou':
                pose_results, next_id = track_iou(pose_results, pose_results_last, next_id, threshold=tracking_thr)
            else:
                for i in range(len(pose_results)):
                    pose_results[i]["track_id"] = i

        # post-process the pose results with smoother
        pose_results = smoother.smooth(pose_results)

        poselist = []
        if pose == 'DEKR' or pose == 'HigherHRNet':
            for i in range(len(pose_results)):
                keypoints = pose_results[i]["keypoints"]
                kptlist = keypoints.tolist()
                poselist.append({"keypoints": kptlist, "track_id": pose_results[i]["track_id"]})
            pose_json.append(poselist)
        else:
            for i in range(len(pose_results)):
                bbox = pose_results[i]["bbox"]
                bboxlist = bbox.tolist()
                keypoints = pose_results[i]["keypoints"]
                kptlist = keypoints.tolist()
                poselist.append({"bbox": bboxlist, "keypoints": kptlist, "track_id": pose_results[i]["track_id"]})
            pose_json.append(poselist)

        # show the results
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame,
            pose_results,
            radius=kpt_radius,
            thickness=line_thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=0.3,
            show=False)

        progress = {"frame_id": frame_id, "image": vis_frame}
        progress_callback.emit(progress)

        if globals.kill_thread:
            return pose_json

    return pose_json


