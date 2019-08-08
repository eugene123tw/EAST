import csv
import glob
import os
import time
from PIL import Image

import cv2
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon

from geometry import *

tf.app.flags.DEFINE_string('training_image_path',
                           '/home/eugene/_DATASETS/scene_text/icdar_2015/train',
                           'training images')

tf.app.flags.DEFINE_string('training_label_path',
                           '/home/eugene/_DATASETS/scene_text/icdar_2015/train_gt',
                           'training labels')

tf.app.flags.DEFINE_string('testing_image_path',
                           '/home/eugene/_DATASETS/scene_text/icdar_2015/test',
                           'testing images')

tf.app.flags.DEFINE_string('testing_label_path',
                           '/home/eugene/_DATASETS/scene_text/icdar_2015/test_gt',
                           'testing labels')

tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_image_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    hard_text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                hard_text_tags.append(True)
            else:
                hard_text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(hard_text_tags, dtype=np.bool)


def crop_area(img, polys, hard_text_tags, crop_background=False, max_tries=50):
    """ Make random crop from the input image.

    Args:
        img (np.ndarray):
        polys (np.ndarray):
            quadrangle ground truths [[x1, y1, x2, y2, x3, y3, x4, y4], [x1, y1, x2, y2, x3, y3, x4, y4]...]
        hard_text_tags (np.ndarray): 1d boolean vector of hard samples
        crop_background (bool):
        max_tries (int): max try of cropping

    Returns:
        img: return cropped background
        polys:
        hard_text_tags:
    """

    h, w, _ = img.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
        w_array[min_x + pad_w: max_x + pad_w] = 1

        min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])
        h_array[min_y + pad_h: max_y + pad_h] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return img, polys, hard_text_tags

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)

        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)

        # area too small
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            continue

        selected_polys = []
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]

        # no text in this area
        if len(selected_polys) == 0:
            if crop_background:
                return img[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], hard_text_tags[selected_polys]
            continue

        # text in this area
        img = img[ymin:ymax + 1, xmin:xmax + 1, :]  # cropping
        polys = polys[selected_polys]
        hard_text_tags = hard_text_tags[selected_polys]
        polys[:, :, 0] -= xmin  # shift x-axis of bounding box after cropping
        polys[:, :, 1] -= ymin  # shift y-axis of bounding box after cropping
        return img, polys, hard_text_tags

    return img, polys, hard_text_tags


def restore_rectangle_rbox(coords, flatten_geo_map):
    """

    Args:
        coords (np.ndarray): (x, y) coordinate where confidence is larger than threshold (default: 0.8)
        flatten_geo_map (np.ndarray):
            Values are from RBOX geometry map where score > threshold.
            The 5 channels represent the sequence as (top, right, bottom, left, angle).
            The first 4 channels are the distances of each pixel to rectangle boundaries
    Returns:

    """
    distances = flatten_geo_map[:, :4]
    angles = flatten_geo_map[:, 4]

    # for angle > 0 ===========================
    coords_0 = coords[angles >= 0]
    d_0 = distances[angles >= 0]
    angle_0 = angles[angles >= 0]

    # boxes that have angle > 0
    if coords_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]]) # center coordinates
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = coords_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        boxes_angle_positive = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        boxes_angle_positive = np.zeros((0, 4, 2))
    # ===========================================

    # boxes that have angle < 0
    # for angle < 0 =============================
    coords_1 = coords[angles < 0]
    d_1 = distances[angles < 0]
    angle_1 = angles[angles < 0]
    if coords_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = coords_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        boxes_angle_negative = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        boxes_angle_negative = np.zeros((0, 4, 2))
    # ==========================================

    # stack all boxes
    return np.concatenate([boxes_angle_positive, boxes_angle_negative])


def generate_masks(img_size, polys, hard_text_tags):
    """

    Args:
        img_size (tuple): (height, width, channel)
        polys:
        hard_text_tags:

    Returns:
        return polygon score map, geometric map and training_mask
    """

    h, w, c = img_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)

    # mask used during training, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)

    for poly_idx, (poly, hard_text) in enumerate(zip(polys, hard_text_tags)):

        # compute a reference length ri for each vertex pi as
        r = [min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]), np.linalg.norm(poly[i] - poly[(i - 1) % 4])) for i in
             range(4)]

        # assign shrink polygon to score map as 1
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)

        # get polygon label map
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

        # if the poly is too small or if the word is too blur, then ignore it in training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))

        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        if hard_text:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):  # top, right, bottom, left
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])

            # ========================================================================
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # parallel lines through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # parallel lines through p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            # ========================================================================

            # ========================================================================
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            # ========================================================================

            # ========================================================================
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # ========================================================================

        # select the smallest parallelogram
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)

        # sort polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def pad_image(img, input_size):
    # pad the image to the training input size or the longer side of image
    h, w, _ = img.shape
    max_side = np.max([h, w, input_size])
    img_padded = np.zeros((max_side, max_side, 3), dtype=np.uint8)
    img_padded[:h, :w, :] = img.copy()
    return img_padded


def generator(input_size=512,
              batch_size=32,
              background_ratio=3. / 8,
              # background_ratio=0,
              random_scale=[0.5, 1, 2.0, 3.0],
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(image_list.shape[0], FLAGS.training_image_path))
    index = np.arange(0, image_list.shape[0])

    while True:

        np.random.shuffle(index)
        padded_images, fnames, score_maps, geo_maps, training_masks = [], [], [], [], []

        for i in index:
            fname = image_list[i]
            img = cv2.imread(fname)

            h, w, _ = img.shape
            gt_fname = os.path.join(FLAGS.training_label_path,
                                    'gt_' + os.path.basename(fname).split('.')[0] + '.txt')
            if not os.path.exists(gt_fname):
                print('Ground truth text file {} does not exists'.format(gt_fname))
                continue

            text_polys, hard_text_tags = load_annoataion(gt_fname)
            text_polys, hard_text_tags = check_and_validate_polys(text_polys, hard_text_tags, (h, w))

            # random scale this image
            rd_scale = np.random.choice(random_scale)
            img = cv2.resize(img, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

            # random crop a area from image
            if np.random.rand() < background_ratio:

                # crop background
                img, text_polys, hard_text_tags = crop_area(img, text_polys, hard_text_tags, crop_background=True)

                # cannot find background
                if text_polys.shape[0] > 0:
                    continue

                # pad and resize image
                new_h, new_w, _ = img.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                padded_img = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                padded_img[:new_h, :new_w, :] = img.copy()
                padded_img = cv2.resize(padded_img, dsize=(input_size, input_size))
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)  # only support RBOX
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            else:
                img, text_polys, hard_text_tags = crop_area(img, text_polys, hard_text_tags, crop_background=False)

                if text_polys.shape[0] == 0:
                    continue

                padded_img = pad_image(img, input_size)

                # resize the image to input size
                h, w, _ = padded_img.shape
                padded_img = cv2.resize(padded_img, dsize=(input_size, input_size))
                resize_w_ratio = input_size / float(w)
                resize_h_ratio = input_size / float(h)
                text_polys[:, :, 0] *= resize_w_ratio
                text_polys[:, :, 1] *= resize_h_ratio
                score_map, geo_map, training_mask = generate_masks(padded_img.shape, text_polys, hard_text_tags)

            if vis:
                fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                axs[0, 0].imshow(padded_img[:, :, ::-1])
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                for poly in text_polys:
                    poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    axs[0, 0].add_artist(Patches.Polygon(
                        poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                axs[0, 1].imshow(score_map[::, ::])
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[1, 0].imshow(geo_map[::, ::, 0])
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(geo_map[::, ::, 1])
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[2, 0].imshow(geo_map[::, ::, 2])
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].imshow(training_mask[::, ::])
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])
                plt.tight_layout()
                plt.show()
                plt.close()

            padded_images.append(padded_img[:, :, ::-1].astype(np.float32))
            fnames.append(fname)
            score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
            geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
            training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

            if len(padded_images) == batch_size:
                yield np.stack(padded_images), \
                      np.stack(fnames), \
                      np.stack(score_maps), \
                      np.stack(geo_maps), \
                      np.stack(training_masks)
                padded_images, fnames, score_maps, geo_maps, training_masks = [], [], [], [], []


if __name__ == '__main__':
    loader = generator(input_size=512, batch_size=8, vis=True)
    print(next(loader))
