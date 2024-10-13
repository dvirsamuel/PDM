# Code from https://github.com/Ali2500/BURST-benchmark
from typing import Optional, List, Tuple, Union, Dict, Any

import cv2
import numpy as np
import os.path as osp

import utils
from PerMIRS.visualization_utils import bbox_from_mask


class BURSTVideo:
    def __init__(self, video_dict: Dict[str, Any], images_dir: Optional[str] = None):

        self.annotated_image_paths: List[str] = video_dict["annotated_image_paths"]
        self.all_images_paths: List[str] = video_dict["all_image_paths"]
        self.segmentations: List[Dict[int, Dict[str, Any]]] = video_dict["segmentations"]
        self._track_category_ids: Dict[int, int] = video_dict["track_category_ids"]
        self.image_size: Tuple[int, int] = (video_dict["height"], video_dict["width"])

        self.id = video_dict["id"]
        self.dataset = video_dict["dataset"]
        self.name = video_dict["seq_name"]
        self.negative_category_ids = video_dict["neg_category_ids"]
        self.not_exhaustive_category_ids = video_dict["not_exhaustive_category_ids"]

        self._images_dir = images_dir

    @property
    def num_annotated_frames(self) -> int:
        return len(self.annotated_image_paths)

    @property
    def num_total_frames(self) -> int:
        return len(self.all_images_paths)

    @property
    def image_height(self) -> int:
        return self.image_size[0]

    @property
    def image_width(self) -> int:
        return self.image_size[1]

    @property
    def track_ids(self) -> List[int]:
        return list(sorted(self._track_category_ids.keys()))

    @property
    def track_category_ids(self) -> Dict[int, int]:
        return {
            track_id: self._track_category_ids[track_id]
            for track_id in self.track_ids
        }

    def filter_dataset_for_benchmark(self, masks_per_frame):
        # First, filter low quality frames and frames that do not contain more than two different instances of the same object
        global_relevant_tracks = [key for key, value in self.track_category_ids.items() if
                                  list(self.track_category_ids.values()).count(value) > 1]
        tracks_count = dict.fromkeys(global_relevant_tracks, 0)
        #tracks_bbox_size_per_frame = dict.fromkeys(global_relevant_tracks, list())
        tracks_per_frame = dict.fromkeys(global_relevant_tracks, list())
        # For each frame, first check if the frame contains multi-instance of the same object, then, proceed to
        # calculate #occurance for each track-id and it's size.
        for i in range(len(masks_per_frame)):
            curr_id_categories = dict(zip(self.segmentations[i].keys(),
                                          list(map(self.track_category_ids.get, self.segmentations[i].keys()))))
            curr_relevant_tracks = [key for key, value in curr_id_categories.items() if
                                    list(curr_id_categories.values()).count(value) > 1]
            if len(curr_relevant_tracks) >= 2:
                #potential_frames += [i]
                frame_masks = masks_per_frame[i]
                frame_size = list(frame_masks.values())[0].size
                tracks_count.update(
                    zip(curr_relevant_tracks, list(map(lambda x: tracks_count.get(x) + 1, curr_relevant_tracks))))
                for t_id in curr_relevant_tracks:
                    xmin, ymin, xmax, ymax = [int(x) for x in bbox_from_mask(frame_masks[t_id])]
                    relative_size = (abs(ymax - ymin) * abs(xmax - xmin) / frame_size)
                    #relative_size = frame_masks[t_id].sum() / frame_size
                    tracks_per_frame[t_id] = tracks_per_frame[t_id] + [(i, relative_size)]
                    #tracks_per_frame.update(
                    #    zip(curr_relevant_tracks,
                    #        list(map(lambda x: tracks_per_frame.get(x) + [(i, relative_size)], curr_relevant_tracks))))
                    #tracks_bbox_size_per_frame[t_id] = tracks_bbox_size_per_frame[t_id] + [
                    #    (abs(ymax - ymin) * abs(xmax - xmin) / frame_size)]

        # Take the instance which appeared the most during the video
        max_inst = max(tracks_count, key=tracks_count.get)
        # Select n frames, where the instance appeared the largest, make sure the frames are far apart.
        inst_size_per_frame = dict(tracks_per_frame[max_inst])
        sorted_frames = np.array(list(dict(sorted(inst_size_per_frame.items(), key=lambda item: item[1], reverse=True)).keys()))
        first_frame = sorted_frames[0]
        final_frames = [first_frame] + list(sorted_frames[np.where(abs(sorted_frames - first_frame) > 15)[0][:2]])
        final_masks = []
        for f in final_frames:
            final_masks += [{max_inst: masks_per_frame[f][max_inst]}]
        return final_frames, final_masks

    def get_image_paths(self, frame_indices: Optional[List[int]] = None) -> List[str]:
        """
        Get file paths to all image frames
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of file paths
        """
        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                                                                                     f"invalid"

        return [osp.join(self._images_dir, self.annotated_image_paths[t]) for t in frame_indices]

    def load_images(self, frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Load annotated image frames for the video
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of images as numpy arrays of dtype uint8 and shape [H, W, 3] (RGB)
        """
        assert self._images_dir is not None, f"Images cannot be loaded because 'images_dir' is None"

        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                                                                                     f"invalid"

        images = []

        for t in frame_indices:
            filepath = osp.join(self._images_dir, self.annotated_image_paths[t])
            assert osp.exists(filepath), f"Image file not found: '{filepath}'"
            images.append(cv2.imread(filepath, cv2.IMREAD_COLOR)[:, :, ::-1])  # convert BGR to RGB

        return images

    def images_paths(self, frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Load annotated image frames for the video
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of images as numpy arrays of dtype uint8 and shape [H, W, 3] (RGB)
        """
        assert self._images_dir is not None, f"Images cannot be loaded because 'images_dir' is None"

        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                                                                                     f"invalid"

        paths = []

        for t in frame_indices:
            filepath = osp.join(self._images_dir, self.annotated_image_paths[t])
            assert osp.exists(filepath), f"Image file not found: '{filepath}'"
            #images.append(cv2.imread(filepath, cv2.IMREAD_COLOR)[:, :, ::-1])  # convert BGR to RGB
            paths += [filepath]

        return paths

    def load_masks(self, frame_indices: Optional[List[int]] = None) -> List[Dict[int, np.ndarray]]:
        """
        Decode RLE masks into mask images
        :param frame_indices: Optional argument specifying list of frame indices to load. All indices should be satisfy
        0 <= t < len(self.num_annotated_frames)
        :return: List of dicts (one per frame). Each dict has track IDs as keys and mask images as values.
        """
        if frame_indices is None:
            frame_indices = list(range(self.num_annotated_frames))
        else:
            assert all([0 <= t < self.num_annotated_frames for t in frame_indices]), f"One or more frame indices are " \
                                                                                     f"invalid"

        zero_mask = np.zeros(self.image_size, bool)
        masks = []

        for t in frame_indices:
            masks_t = dict()

            for track_id in self.track_ids:
                if track_id in self.segmentations[t]:
                    masks_t[track_id] = utils.rle_ann_to_mask(self.segmentations[t][track_id]["rle"], self.image_size)
                else:
                    masks_t[track_id] = zero_mask

            masks.append(masks_t)

        return masks

    def filter_category_ids(self, category_ids_to_keep: List[int]):
        track_ids_to_keep = [
            track_id for track_id, category_id in self._track_category_ids.items()
            if category_id in category_ids_to_keep
        ]

        self._track_category_ids = {
            track_id: category_id for track_id, category_id in self._track_category_ids.items()
            if track_id in track_ids_to_keep
        }

        for t in range(self.num_annotated_frames):
            self.segmentations[t] = {
                track_id: seg for track_id, seg in self.segmentations[t].items()
                if track_id in track_ids_to_keep
            }

    def stats(self) -> Dict[str, Any]:
        total_masks = 0
        for segs_t in self.segmentations:
            total_masks += len(segs_t)

        return {
            "Annotated frames": self.num_annotated_frames,
            "Object tracks": len(self.track_ids),
            "Object masks": total_masks,
            "Unique category IDs": list(set(self.track_category_ids.values()))
        }

    def load_first_frame_annotations(self) -> List[Dict[int, Dict[str, Any]]]:
        annotations = []
        for t in range(self.num_annotated_frames):
            annotations_t = dict()

            for track_id, annotation in self.segmentations[t].items():
                annotations_t[track_id] = {
                    "mask": utils.rle_ann_to_mask(annotation["rle"], self.image_size),
                    "bbox": annotation["bbox"],  # xywh format
                    "point": annotation["point"]
                }

            annotations.append(annotations_t)

        return annotations
