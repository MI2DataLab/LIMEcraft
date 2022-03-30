#!/usr/bin/env python
# coding: utf-8


import os
import copy
from functools import partial
from operator import itemgetter

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm


from lime_library.lime import lime_base
from lime_library.lime import lime_image
from lime_library.lime.wrappers.scikit_image import SegmentationAlgorithm

import matplotlib.pyplot as plt

import cv2
import skimage
from skimage.segmentation import mark_boundaries
from PIL import Image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions


### Function from file squares_and_prediction.ipynb

class ImageExplanation(object):
    def __init__(self, image, segments):
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask



def create_mask(img):
    """
    create mask for the image
    """
    # Parameters
    height = img.shape[1]
    width = img.shape[0]
    size = 70
    
    # Create the basic mask
    mask = np.zeros(shape=[height, width], dtype="int")
        
    xs = [0, size, 4*size, 3*size, size, size]
    ys = [0, 0, 2*size, 3*size, size, 3*size]
    
    # Draw a filled rectangle on the mask image
    for i in range(len(xs)): 
        x = xs[i]
        y = ys[i]
        
        rr, cc = skimage.draw.rectangle(start=(y, x),  extent=(min(size, height-y), min(size,width-x)))
        mask[rr, cc] = 1+i
    
    return mask




class LimeImageExplainer(object):
    
    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
    
    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segments=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True,
                         inner_n_segments=0,
                         outer_n_segments=0):
        """
        segments = mask
        """

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)


        if segments is None:
            segments = create_mask(image)
            
# -----------------

        if inner_n_segments>0 or outer_n_segments>0:
            segments_bin = np.copy(segments) # for segmenting the exterior of marked areas
            if outer_n_segments > 0:
                segments_bin[segments_bin!=0]=1
                segments_bin = np.array(segments_bin, dtype=bool)
                segments_bin = np.invert(segments_bin)
                segments_bin = np.array(segments_bin, dtype=int)
                segmentation_fn = SegmentationAlgorithm('slic', n_segments=outer_n_segments, compactness=10, mask=segments_bin)
                segments_bin = segmentation_fn(image)

            if inner_n_segments > 0:     
                # for segmenting the interior of marked areas
                segmentation_fn = SegmentationAlgorithm('slic', n_segments=inner_n_segments, compactness=10, mask=segments)
                segments = segmentation_fn(image)

            if inner_n_segments>0 and outer_n_segments > 0:
                # applying masks
                for i in range(segments.shape[0]):
                    for j in range(segments.shape[1]):
                        if segments[i][j] == 0:
                            segments[i][j]=segments_bin[i][j] + inner_n_segments + 1
    
        plt.imshow(segments)
        plt.axis('off')
        plt.show()
# -----------------


        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp



    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):

        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)



def transform_img_fn(img_list, model):
    """
    img_list - list of np.array
    """
    out = []
    for img in img_list:
        img = image.array_to_img(img)
        img = img.resize(model.input_shape)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = model.inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)



def test_lime(model, img):
    """
    img - array type
    Tests the performance of the algorithm for the top 1 class label
    """
        
    images = transform_img_fn([img], model)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), model.inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)
    segments = explanation.segments
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    out2 = mark_boundaries(mark_boundaries(temp / 2 + 0.5, mask), segments)

    return [out2, mask]



def test_limecraft(model, img, mask, inner_n_segments, outer_n_segments):
    """
    img - array type
    Tests the performance of the algorithm for the top 1 class label
    """
    
    images = transform_img_fn([img], model)
    preds = model.inet_model.predict(images)
    
    decoded_predictions = None
    if model.target_names:
        decoded_predictions = [list(x) for x in zip(list(range(len(model.target_names))), model.target_names, preds[0])]
        decoded_predictions = sorted(decoded_predictions, key=itemgetter(2), reverse=True)

    else:
        decoded_predictions=decode_predictions(preds)[0]

    img = images[0]
    
    img_double = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX) # double
    
    # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
    explainer = LimeImageExplainer()

    explanation = explainer.explain_instance(img_double, model.inet_model.predict, top_labels=5, num_samples=1000, segments=mask,
                                            inner_n_segments=inner_n_segments, outer_n_segments=outer_n_segments)
    
    temp, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
    
    segments = explanation.segments
    out1 = mark_boundaries(mark_boundaries(temp, mask2), segments)

    return [out1, decoded_predictions, mask2]
