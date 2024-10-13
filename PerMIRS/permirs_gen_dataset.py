import shutil
from argparse import ArgumentParser
from tqdm import tqdm
from dataset import BURSTDataset
import numpy as np
import os
import os.path as osp


def main(args):
    dataset = BURSTDataset(annotations_file=args.annotations_file,
                           images_base_dir=args.images_base_dir)

    for i in tqdm(range(dataset.num_videos)):
        try:
            video = dataset[i]

            if args.seq and f"{video.dataset}/{video.name}" != args.seq:
                continue

            print(f"- Dataset: {video.dataset}\n"
                  f"- Name: {video.name}")

            if args.first_frame_annotations:
                annotations = video.load_first_frame_annotations()
            else:
                annotations = video.load_masks()

            frames_idx, annotations = video.filter_dataset_for_benchmark(annotations)
            image_paths = video.images_paths(frames_idx)
            if len(image_paths) < 3:
                raise Exception(i, "not enough images found")

            base_path = f"/PerMIRS/{i}"
            for f_idx, image_p in enumerate(image_paths):
                frame_p = base_path + f"/{f_idx}.{osp.split(image_p)[-1].split('.')[-1]}"
                os.makedirs(os.path.dirname(frame_p), exist_ok=True)
                shutil.copyfile(image_p, frame_p)

            np.save(base_path + "/masks.npz", annotations)
        except Exception as e:
            print(i, e)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--images_base_dir", required=True)
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--first_frame_annotations", action='store_true')

    # extra options
    parser.add_argument("--save_dir", required=False)
    parser.add_argument("--seq", required=False)

    main(parser.parse_args())