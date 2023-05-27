
from typing import Any
import tensorflow as tf
import numpy as np
from datasets import load_dataset

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE=4

def get_random_crop(image_dim):
    def random_crop(image):
        cropped_image = tf.image.random_crop(
        image, size=[image_dim, image_dim, 3])
        return cropped_image
    return random_crop

def get_normalize(image_dim,method):
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        image = tf.image.resize(image, [image_dim, image_dim],
                                method=method)
        return image
    return normalize

def denormalize(norm_image):
    return np.uint8(127.5*norm_image+127.5)
    

def get_random_jitter(image_dim):
    random_crop=get_random_crop(image_dim)
    def random_jitter(image):
        # resizing to 286 x 286 x 3
        resized_dim=int(1.1*image_dim)
        image = tf.image.resize(image, [resized_dim, resized_dim],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image
    return random_jitter

def get_reprocess_image_train(image_dim,method):
    def preprocess_image_train(image):
        random_jitter=get_random_jitter(image_dim)
        normalize=get_normalize(image_dim)
        image = normalize(image,method)
        image = random_jitter(image)
        return image
    return preprocess_image_train

def equal_length(smaller, bigger):
    if len(bigger) < len(smaller):
        bigger, new_smaller = equal_length(bigger,smaller)
    else:
        new_smaller=[]
        for i in range(len(bigger)):
            new_smaller.append(smaller[i%len(smaller)])
    return new_smaller, bigger

class OneHotEncoder:
    def __init__(self,path_list):
        self.path_list=path_list

    def __call__(self, path):
        index=self.path_list.index(path)
        ret=[0 for _ in self.path_list]
        ret[index]=1
        #print(index,path,ret)
        return tf.cast(ret, tf.float32)

def get_labeled_datasets_train(path_list,image_dim, preprocess=False,method=tf.image.ResizeMethod.GAUSSIAN):
    labels=[]
    images=[]
    onehot=OneHotEncoder(path_list)
    for path in path_list:
        path_images=[np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
        path_labels=[onehot(path) for _ in path_images]
        images+=path_images
        labels+=path_labels
    if preprocess:
        preprocess_image_train=get_reprocess_image_train(image_dim,method)
        image_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE)
    else:
        normalize=get_normalize(image_dim,method)
        image_dataset= tf.data.Dataset.from_tensor_slices(images).map(
            normalize, num_parallel_calls=AUTOTUNE)
    label_dataset=tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((image_dataset, label_dataset))

def get_labeled_datasets_test(path_list,image_dim, method=tf.image.ResizeMethod.GAUSSIAN):
    labels=[]
    images=[]
    onehot=OneHotEncoder(path_list)
    for path in path_list:
        path_images=[np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='test']
        path_labels=[onehot(path) for _ in path_images]
        images+=path_images
        labels+=path_labels
    normalize=get_normalize(image_dim,method)
    image_dataset= tf.data.Dataset.from_tensor_slices(images).map(
        normalize, num_parallel_calls=AUTOTUNE)
    label_dataset=tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((image_dataset, label_dataset))


def get_single_dataset_train(unit_test,path,image_dim, preprocess=False,method=tf.image.ResizeMethod.GAUSSIAN):
    if unit_test:
        data_frame= [np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='train']
    else:
        data_frame= [np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
    if preprocess:
        preprocess_image_train=get_reprocess_image_train(image_dim,method)
        return tf.data.Dataset.from_tensor_slices(data_frame).map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE)
    else:
        normalize=get_normalize(image_dim,method)
        return tf.data.Dataset.from_tensor_slices(data_frame).map(
        normalize, num_parallel_calls=AUTOTUNE)
    
def get_single_dataset_test(unit_test,path,image_dim,method=tf.image.ResizeMethod.GAUSSIAN):
    if unit_test:
        data_frame= [np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='test']
    else:
        data_frame= [np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='test']
    normalize=get_normalize(image_dim, method)
    return tf.data.Dataset.from_tensor_slices(data_frame).map(
    normalize, num_parallel_calls=AUTOTUNE)

def get_multiple_datasets_train(path_list,image_dim,preprocess=False,method=tf.image.ResizeMethod.GAUSSIAN):
    images=[]
    for path in path_list:
        path_images=[np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
        images+=path_images
    if preprocess:
        preprocess_image_train=get_reprocess_image_train(image_dim,method)
        image_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            preprocess_image_train, num_parallel_calls=AUTOTUNE)
    else:
        normalize=get_normalize(image_dim,method)
        image_dataset= tf.data.Dataset.from_tensor_slices(images).map(
            normalize, num_parallel_calls=AUTOTUNE)
    return image_dataset

def get_multiple_datasets_test(path_list,image_dim,method=tf.image.ResizeMethod.GAUSSIAN):
    images=[]
    for path in path_list:
        path_images=[np.array(img['image']) for img in load_dataset(path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
        images+=path_images
    normalize=get_normalize(image_dim,method)
    image_dataset= tf.data.Dataset.from_tensor_slices(images).map(
        normalize, num_parallel_calls=AUTOTUNE)
    return tf.data.Dataset.from_tensor_slices(image_dataset)



def get_datasets_train(unit_test,content_path, style_path,image_dim, preprocess=False,method=tf.image.ResizeMethod.GAUSSIAN):
    if unit_test:
        data_frame_style= [np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='train']
        data_frame_content=[np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='train']
    else:
        data_frame_style= [np.array(img['image']) for img in load_dataset(style_path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
        data_frame_content=[np.array(img['image']) for img in load_dataset(content_path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='train']
    data_frame_style, data_frame_content = equal_length(data_frame_style, data_frame_content)
    if preprocess:
        img_function=get_reprocess_image_train(image_dim,method)
    else:
        img_function=get_normalize(image_dim, method)
    train_style= tf.data.Dataset.from_tensor_slices(data_frame_style).map(
        img_function, num_parallel_calls=AUTOTUNE)
    train_content = tf.data.Dataset.from_tensor_slices(data_frame_content).map(
        img_function, num_parallel_calls=AUTOTUNE)
    return train_style, train_content

def get_datasets_test(unit_test,content_path, style_path,image_dim,method=tf.image.ResizeMethod.GAUSSIAN):
    if unit_test:
        data_frame_style= [np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='test']
        data_frame_content=[np.array(img['image']) for img in load_dataset("jlbaker361/little_dataset",split="train") if img['split']=='test']
    else:
        data_frame_style= [np.array(img['image']) for img in load_dataset(style_path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='test']
        data_frame_content=[np.array(img['image']) for img in load_dataset(content_path,split="train",cache_dir="../../../../../scratch/jlb638/hf_cache") if img['split']=='test']
    data_frame_style, data_frame_content = equal_length(data_frame_style, data_frame_content)
    normalize=get_normalize(image_dim,method)
    test_style= tf.data.Dataset.from_tensor_slices(data_frame_style).map(
        normalize, num_parallel_calls=AUTOTUNE)
    test_content = tf.data.Dataset.from_tensor_slices(data_frame_content).map(
        normalize, num_parallel_calls=AUTOTUNE)
    return test_style, test_content