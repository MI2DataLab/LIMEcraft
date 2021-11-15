import numpy as np
from skimage import draw
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from skimage.restoration import inpaint
from ast import literal_eval



## Support functions

def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype('int')



def path_to_mask(path, shape, relayout_data):
    """
    Take the paths to all the areas and create a mask from scratch with all these areas.
    """
    mask = np.zeros([shape[0], shape[1]], dtype="int") #mask base

    iter = 1 # value for a given area
    for shape_info in relayout_data["shapes"]:
        path = shape_info["path"]
        cols, rows = path_to_indices(path).T
        rr, cc = draw.polygon(rows, cols)
        rr = np.clip(rr, 0, shape[0]-1)
        cc = np.clip(cc, 0, shape[1]-1)
        mask[rr, cc] = iter 
        iter+=1
    
    return mask



def transform_preds(preds_arr):
    """
    From the prediction result, it leaves only the relevant information - the label and the percentage of confidence
    """
    elegant_preds = []
    for pred in preds_arr:
        elegant_preds.append([pred[1], np.round(np.double(pred[2])*100,2)])
    return elegant_preds



def merge_df_str(str1, str2):
    """
    Inner join - for comparing prediction before and after image editing
    """
    df1 = pd.DataFrame(transform_preds(literal_eval(str1)), columns=["class", "%_original_img"])
    df2 = pd.DataFrame(transform_preds(literal_eval(str2)), columns=["class", "%_edited_img"])
    return pd.merge(df1, df2, on="class", how='outer').fillna('-')



def get_segment_crop(img,tol=0, mask=None):
    """
    Cuts a portion of the photo based on the mask
    """
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]



def find_centrum(path):
    points = path_to_indices(path)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    (x_centrum, y_centrum) = (int(np.round(sum(x)/len(points))), int(np.round(sum(y) / len(points))))
    return (x_centrum, y_centrum)



def circle_rotate(im, x, y, radius, degree, mask, sub_mask=None, left_right=0, up_down=0):
    """
    Rotates and shifts the selected part of the picture

    """
    img_arr = np.array(im)
    box = (max(x-radius,0), max(y-radius,0), min(x+radius+1,224), min(y+radius+1,224))
    crop = im.crop(box=box)
    crop_arr = np.asarray(crop)
    # build the circle mask
    if sub_mask is None:
        sub_mask = create_mask(np.array(im.resize((2*radius+1, 2*radius+1))))

    # create the new circular image
    (s1,s2,s3) = crop_arr.shape
    sub_img_arr = np.empty((s1,s2,4) ,dtype='uint8')
    sub_img_arr[:,:,:3] = crop_arr[:,:,:3]
    sub_img_arr[:,:,3] = sub_mask*255

    sub_img = Image.fromarray(sub_img_arr, "RGBA").rotate(degree)
    plt.imshow(sub_img)

    i2 = inpaint.inpaint_biharmonic(img_arr, mask, multichannel=True) # inpainting
    i2 = image.array_to_img(i2)
    i2.paste(sub_img, (box[0]+left_right, box[1]-up_down), sub_img.convert('RGBA'))
    return i2



# Expanding/shrinking the area
def remapping(im, path, power):
    """
    Expands the fragment with the selected center and radius.
    power - if >1.0 for expansion, <1.0 for shrinkage
    """
    img_arr = np.array(im)
    
    # look for the center of gravity of the selected area
    (x_centrum, y_centrum) = find_centrum(path)
    
#     https://stackoverflow.com/questions/58237736/how-to-do-deformations-in-opencv
    height, width, _ = img_arr.shape
    map_y = np.zeros((height,width),dtype=np.float32)
    map_x = np.zeros((height,width),dtype=np.float32)

    # create index map
    for i in range(height):
        for j in range(width):
            map_y[i][j]=i
            map_x[i][j]=j

    

    points = path_to_indices(path)
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # look for protruding points to draw a rectangle that covers the entire area
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    radius = np.max([x_centrum-x_min, x_max-x_centrum, 
                    y_centrum-y_min, y_max-y_centrum])
        
    # deform around the right eye
    for i in range (-radius, radius): # OY axis
        for j in range(-radius, radius): # OX axis
            point = Point(x_centrum+j,y_centrum+i)
            polygon = Polygon(points)
            if not polygon.contains(point):
                continue

            if i > 0:
                map_y[y_centrum + i][x_centrum + j] = y_centrum + (i/radius)**power * radius
            if i < 0:
                map_y[y_centrum + i][x_centrum + j] = y_centrum - (-i/radius)**power * radius
            if j > 0:
                map_x[y_centrum + i][x_centrum + j] = x_centrum + (j/radius)**power * radius
            if j < 0:
                map_x[y_centrum + i][x_centrum + j] = x_centrum - (-j/radius)**power * radius

    warped=cv2.remap(np.float32(im),map_x,map_y,cv2.INTER_LINEAR)
    return warped



def merge_all_preds(str_original, str_color, str_rotation, str_shape):
    """
    Internal join - to report
    """
    if str_original is not None:
        df1 = pd.DataFrame(transform_preds(literal_eval(str_original)), columns=["class", "%_original_img"])
    else:
        df1 = pd.DataFrame(columns=["class", "%_original_img"])
    if str_color is not None:
        df2 = pd.DataFrame(transform_preds(literal_eval(str_color)), columns=["class", "%_color_img"])
    else:
        df2 = pd.DataFrame(columns=["class", "%_color_img"])
    if str_rotation is not None:
        df3 = pd.DataFrame(transform_preds(literal_eval(str_rotation)), columns=["class", "%_rotation_img"])
    else:
        df3 = pd.DataFrame(columns=["class", "%_rotation_img"])
    if str_shape is not None:
        df4 = pd.DataFrame(transform_preds(literal_eval(str_shape)), columns=["class", "%_shape_img"])
    else:
        df4 = pd.DataFrame(columns=["class", "%_shape_img"])
    

    df=pd.merge(df1, df2, on="class", how='outer').fillna('-')
    df=pd.merge(df, df3, on="class", how='outer').fillna('-')
    df=pd.merge(df, df4, on="class", how='outer').fillna('-')

    return df