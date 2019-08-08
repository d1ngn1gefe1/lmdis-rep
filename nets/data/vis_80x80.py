import os
import cv2
import numpy as np
import json
import time
import threading
import random
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
import scipy.io as sio
import importlib.util


class Net:
  def __init__(self, subset_name='train', options=None):

    self._debug = False
    self._shuffle = False
    self._cache_size = 3000
    self._mean_reduce = False
    self._mean = [5.0, 10.0, 15.0]
    if options != None and options != {}:
      if 'cache_size' in options:
        self._cache_size = options['cache_size']
      if 'mean_reduce' in options:
        self._mean_reduce = options['mean_reduce']
      if 'shuffle' in options:
        self._shuffle = options['shuffle']
      if 'debug' in options:
        self._debug = options['debug']

    dataset_dir = '/private/home/alanzluo/data/vos'
    api_file = '/private/home/alanzluo/code/cvpr20/datasets/vis_api.py'
    spec = importlib.util.spec_from_file_location('VisApi', api_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    self.api = module.VisApi(dataset_dir, 'train')  # valid split does not have annotation

    # # make dataset
    # cat_nms = 'zebra'
    # cat_ids = self.api.get_cat_ids(cat_nms=cat_nms)
    # ann_ids = self.api.get_ann_ids(cat_ids=cat_ids)
    # anns = self.api.load_anns(ann_ids)
    # self.dataset = []
    # for ann in anns:
    #   video_id = ann['video_id']
    #   vid = self.api.load_vids(video_id)[0]
    #   for bbox, file_name in zip(ann['bboxes'], vid['file_names']):
    #     if bbox is not None:
    #       self.dataset.append({'bbox': np.array(bbox, dtype=int), 'file_name': file_name})

    # make dataset
    cat_nms = 'zebra'
    cat_ids = self.api.get_cat_ids(cat_nms=cat_nms)
    fanns = self.api.get_fanns(cat_ids=cat_ids, crowd=False, area=[2500, float('inf')], hw_ratio=[0.5, 2],
                               in_frame=True)
    self.dataset = []
    for fann in fanns:
      video_id = fann['video_id']
      vid = self.api.load_vids(video_id)[0]
      self.dataset.append({'bbox': np.array(fann['bbox'], dtype=int), 'file_name': vid['file_names'][fann['frame_id']]})

    self._num_samples = len(self.dataset)
    self._waitlist = list(range(len(self.dataset)))
    if self._shuffle:
      random.shuffle(self._waitlist)
    self._dataset = None
    self._cur_pos = 0  # num of sample done in this epoch
    self._cur_epoch = 0  # current num of epoch
    self._cur_iter = 0  # num of batches returned
    self._num_fields = 1
    self._out_h = 80
    self._out_w = 80

    self._image_cache = []

    self._lock = threading.Lock()

    # self.set_dataset()

    self._pool_size = cpu_count()

    self._pool = Pool(self._pool_size)
    self._cache_thread = threading.Thread(target=self.preload_dataset)
    self._cache_thread.start()

  def read_image(self, i):
    image_file = os.path.join(self.api.vid_dir, self.dataset[i]['file_name'])
    bbox = self.dataset[i]['bbox']

    # The channel for cv2.imread is B, G, R
    if not os.path.exists(image_file):
      print(image_file)
    image_arr = cv2.imread(image_file)
    image_arr = image_arr[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
    image_arr = cv2.resize(image_arr, (100, 100))

    h, w, _ = image_arr.shape
    margin_h = (h-self._out_h)//2
    margin_w = (w-self._out_w)//2
    image_arr = image_arr[margin_h:margin_h+self._out_h, margin_w:margin_w+self._out_w]

    result = image_arr.astype(np.float32)/np.array(255., dtype=np.float32)
    result[:, :, [0, 1, 2]] = result[:, :, [2, 1, 0]]

    return result

  def __call__(self, *args, **kwargs):
    return self.next_batch(*args, **kwargs)

  def num_samples(self):
    return self._num_samples

  def epoch(self):
    return self._cur_epoch

  def iter(self):
    return self._cur_iter

  def num_fields(self):
    return self._num_fields

  def num_samples_finished(self):
    return self._cur_pos

  def reset(self):
    """ Reset the state of the data loader
    E.g., the reader points at the beginning of the dataset again
    :return: None
    """
    self._cur_pos = 0
    self._cur_epoch = 0
    self._cur_iter = 0
    self._waitlist = list(range(len(self.dataset)))
    if self._shuffle:
      random.shuffle(self._waitlist)
    tmp = 0
    while self._cache_thread.isAlive():
      tmp += 1
    self._cache_thread = threading.Thread(target=self.preload_dataset)
    self._lock.acquire()
    self._image_cache = []
    self._lock.release()
    self._cache_thread.start()

  def preload_dataset(self):
    if self._debug:
      print("preload")
    if len(self._image_cache) > self._cache_size:
      return
    else:
      while len(self._image_cache) < 1000:
        if len(self._waitlist) < 1000:
          self._waitlist += list(range(len(self.dataset)))
          if self._shuffle:
            random.shuffle(self._waitlist)

        results = self._pool.map(self.read_image, self._waitlist[:1000])
        del self._waitlist[:1000]
        self._lock.acquire()
        self._image_cache = self._image_cache+list(results)
        self._lock.release()
      if self._debug:
        print(len(self._image_cache))

  def next_batch(self, batch_size):
    """ fetch the next batch
    :param batch_size: next batch_size
    :return: a tuple includes all data
    """
    if batch_size < 0:
      batch_size = 0
    if self._cache_size < 3*batch_size:
      self._cache_size = 3*batch_size

    this_batch = [None]*self._num_fields

    if len(self._image_cache) < batch_size:
      if self._debug:
        print("Blocking!!, Should only appear once with proper setting")

      if not self._cache_thread.isAlive():
        self._cache_thread = threading.Thread(target=self.preload_dataset)
        self._cache_thread.start()
      self._cache_thread.join()

      self._lock.acquire()
      this_batch[0] = self._image_cache[0:batch_size]
      del self._image_cache[0:batch_size]
      self._lock.release()
    else:
      self._lock.acquire()
      this_batch[0] = self._image_cache[0:batch_size]
      del self._image_cache[0:batch_size]
      self._lock.release()
      if not self._cache_thread.isAlive():
        self._cache_thread = threading.Thread(target=self.preload_dataset)
        self._cache_thread.start()

    self._cur_iter += 1
    self._cur_pos = self._cur_pos+batch_size
    if self._cur_pos >= self._num_samples:
      self._cur_epoch += 1
      self._cur_pos = self._cur_pos%self._num_samples

    return this_batch

  @staticmethod
  def output_types():  # only used for net instance
    t = ["float32"]
    return t

  @staticmethod
  def output_shapes():
    t = [(None, 80, 80, 3)]  # None for batch size
    return t

  @staticmethod
  def output_ranges():
    return [1.]

  @staticmethod
  def output_keys():
    return ["data"]


if __name__ == '__main__':
  main()
