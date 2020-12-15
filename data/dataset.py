import os
import pickle

import numpy as np
import torch.utils.data
import tqdm
from torchvision.datasets import DatasetFolder


def npy_loader(path):
    return torch.from_numpy(np.load(path))


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=32):
        dataset = DatasetFolder(folder, npy_loader, extensions=('.npy',))
        self.total_frames = 0
        self.lengths = []
        self.arrays = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'rb') as f:
                self.arrays, self.lengths = pickle.load(f)
        else:
            for idx, (array, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                array_path, _ = dataset.samples[idx]
                length = len(array)
                if length >= min_len:
                    self.arrays.append((array_path, categ))
                    self.lengths.append(length)
            if cache is not None:
                with open(cache, 'wb') as f:
                    pickle.dump((self.arrays, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print(("Total number of frames {}".format(np.sum(self.lengths))))

    def __getitem__(self, item):
        path, label = self.arrays[item]
        arr = np.load(path).astype('float32')
        return arr, label

    def __len__(self):
        return len(self.arrays)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]

        frame = video[frame_num]
        if frame.shape[0] == 0:
            print(("video {}. num {}".format(video.shape, item)))

        return {"images": self.transforms(frame), "categories": target}

    def __len__(self):
        return self.dataset.cumsum[-1]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        video, target = self.dataset[item]

        video_len = video.shape[0]

        # videos can be of various length, we randomly sample sub-sequences
        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            raise Exception("Frame skip is too high id - {}, len - {}, frame skip - {}").format(self.dataset[item],
                                                                                                video_len,
                                                                                                self.every_nth)
        else:
            raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)

        selected = video[subsequence_idx]

        return {"images": self.transforms(selected), "categories": target}

    def __len__(self):
        return len(self.dataset)
