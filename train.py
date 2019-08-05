import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import dataset
import model

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 0, '')
tf.app.flags.DEFINE_float('lr', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('checkpoint_path', '/home/eugene/_MODELS/east_icdar2015_resnet_v1_50_rbox', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

FLAGS = tf.app.flags.FLAGS


def multi_loss(images, score_maps, geo_maps, training_masks):
    """

    Args:
        images:
        score_maps: ground truth score masks
        geo_maps: ground truth RBOX geometric masks
        training_masks:

    Returns:

    """
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope()):
        y_predict_score_masks, y_predict_geometry_masks = model.model(images, is_training=True)

    model_loss = model.loss(score_maps, y_predict_score_masks, geo_maps, y_predict_geometry_masks, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    tensors = {}
    tensors['input'] = images
    tensors['score_map'] = score_maps
    tensors['score_map_pred'] = y_predict_score_masks
    tensors['geo_map'] = geo_maps
    tensors['geo_map_pred'] = y_predict_geometry_masks
    tensors['training_masks'] = training_masks

    tf.summary.image('input', images)
    tf.summary.image('score_map', score_maps)
    tf.summary.image('score_map_pred', y_predict_score_masks * 255)
    tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
    tf.summary.image('geo_map_0_pred', y_predict_geometry_masks[:, :, :, 0:1])
    tf.summary.image('training_masks', training_masks)
    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, tensors


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_geo_maps_ph = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    input_training_masks_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    with tf.device('/gpu:0'):
        with tf.name_scope('model_0') as scope:
            total_loss_op, model_loss_op, tensors = multi_loss(
                input_images_ph,
                input_score_maps_ph,
                input_geo_maps_ph,
                input_training_masks_ph)
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            grads = opt.compute_gradients(total_loss_op)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()

    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = dataset.generator(input_size=FLAGS.input_size, batch_size=FLAGS.batch_size)

        start = time.time()
        for step in range(FLAGS.max_steps):
            images, fnames, score_maps, geo_maps, training_masks = next(data_generator)
            model_loss, total_loss, _ = sess.run([model_loss_op, total_loss_op, train_op],
                                                 feed_dict={input_images_ph: images,
                                                            input_score_maps_ph: score_maps,
                                                            input_geo_maps_ph: geo_maps,
                                                            input_training_masks_ph: training_masks})
            if np.isnan(total_loss):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * FLAGS.batch_size) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, model_loss, total_loss, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, total_loss, summary_str = sess.run([train_op, total_loss_op, summary_op],
                                                      feed_dict={input_images_ph: images,
                                                                 input_score_maps_ph: score_maps,
                                                                 input_geo_maps_ph: geo_maps,
                                                                 input_training_masks_ph: training_masks})
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    tf.app.run()
