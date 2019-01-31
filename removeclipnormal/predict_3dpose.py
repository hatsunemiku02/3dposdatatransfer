
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ntpath
import math
import os
import random
import sys
import time
import h5py
import copy

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import procrustes

import viz
import cameras
import data_utils
import linear_model
import my_model

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 1, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 200, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("camera_frame", False, "Convert 3d poses to camera coordinates")
tf.app.flags.DEFINE_boolean("max_norm", False, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("remove_max_norm", False, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", False, "Use batch_normalization")

# Data loading
tf.app.flags.DEFINE_boolean("predict_14", False, "predict 14 joints")
tf.app.flags.DEFINE_boolean("use_sh", False, "Use 2d pose predictions from StackedHourglass")
tf.app.flags.DEFINE_string("action","All", "The action to train on. 'All' means all the actions")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", False, "Whether to add a residual connection every 2 layers")

# Evaluation
tf.app.flags.DEFINE_boolean("procrustes", False, "Apply procrustes analysis at test time")
tf.app.flags.DEFINE_boolean("evaluateActionWise",False, "The dataset to use either h36m or heva")

# Directories
tf.app.flags.DEFINE_string("cameras_path","data/h36m/cameras.h5","Directory to load camera parameters")
tf.app.flags.DEFINE_string("data_dir",   "data/h36m/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "experiments", "Training directory.")
tf.app.flags.DEFINE_string("mysave_dir", "mysave", "mysave directory.")
# Train or load
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("hanksam", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

# Misc
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join( FLAGS.train_dir,
  FLAGS.action,
  'dropout_{0}'.format(FLAGS.dropout),
  'epochs_{0}'.format(FLAGS.epochs) if FLAGS.epochs > 0 else '',
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual' if FLAGS.residual else 'not_residual',
  'depth_{0}'.format(FLAGS.num_layers),
  'linear_size{0}'.format(FLAGS.linear_size),
  'batch_size_{0}'.format(FLAGS.batch_size),
  'procrustes' if FLAGS.procrustes else 'no_procrustes',
  'maxnorm' if FLAGS.max_norm else 'no_maxnorm',
  'batch_normalization' if FLAGS.batch_norm else 'no_batch_normalization',
  'use_stacked_hourglass' if FLAGS.use_sh else 'not_stacked_hourglass',
  'predict_14' if FLAGS.predict_14 else 'predict_17')

print( train_dir )
mysave_dir = FLAGS.mysave_dir
print( mysave_dir )

summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

# To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, actions, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def create_model_my( session, actions, batch_size ):
  """
  Create model and initialize it or load its parameters in a session

  Args
    session: tensorflow session
    actions: list of string. Actions to train/test on
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """

  model = my_model.MyModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      FLAGS.predict_14,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32,
      FLAGS.remove_max_norm
      )

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  #ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  #print( "train_dir", train_dir )
  ckpt = tf.train.get_checkpoint_state( "gooddata", latest_filename="checkpoint")


  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific checkpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join("gooddata","gogo.index")):
        ckpt_name = os.path.join( os.path.join("gooddata","checkpoint") )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def train():
  """Train a linear model for 3d pose estimation"""

  actions = data_utils.define_actions( FLAGS.action )

  number_of_actions = len( actions )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
  if FLAGS.use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )
  print( "done reading and normalizing data." )

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto(
    device_count=device_count,
    allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("Creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, actions, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("Model created")

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100

    for _ in xrange( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs = model.get_all_batches( train_set_2d, train_set_3d, FLAGS.camera_frame, training=True )
      nbatches = len( encoder_inputs )
      print("There are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.

      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}... ".format( current_epoch, i+1, nbatches), end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =  model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batches
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False

      if FLAGS.evaluateActionWise:

        print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs

        cum_err = 0
        for action in actions:

          print("{0:<12} ".format(action), end="")
          # Get 2d and 3d testing data for this action
          action_test_set_2d = get_action_subset( test_set_2d, action )
          action_test_set_3d = get_action_subset( test_set_3d, action )
          encoder_inputs, decoder_outputs = model.get_all_batches( action_test_set_2d, action_test_set_3d, FLAGS.camera_frame, training=False)

          act_err, _, step_time, loss = evaluate_batches( sess, model,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
            current_step, encoder_inputs, decoder_outputs )
          cum_err = cum_err + act_err

          print("{0:>6.2f}".format(act_err))

        summaries = sess.run( model.err_mm_summary, {model.err_mm: float(cum_err/float(len(actions)))} )
        model.test_writer.add_summary( summaries, current_step )
        print("{0:<12} {1:>6.2f}".format("Average", cum_err/float(len(actions) )))
        print("{0:=^19}".format(''))

      else:

        n_joints = 17 if not(FLAGS.predict_14) else 14
        encoder_inputs, decoder_outputs = model.get_all_batches( test_set_2d, test_set_3d, FLAGS.camera_frame, training=False)

        total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
          data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
          data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
          current_step, encoder_inputs, decoder_outputs, current_epoch )

        print("=============================\n"
              "Step-time (ms):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % ( 1000*step_time, loss, total_err ))

        for i in range(n_joints):
          # 6 spaces, right-aligned, 5 decimal places
          print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
        print("=============================")

        # Log the error to tensorboard
        summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
        model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}


def evaluate_batches( sess, model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if not(FLAGS.predict_14) else 14
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(FLAGS.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]

    assert dec_out.shape[0] == FLAGS.batch_size
    assert poses3d.shape[0] == FLAGS.batch_size

    if FLAGS.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(FLAGS.batch_size):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c

        poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(FLAGS.predict_14) else np.reshape(out,[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == FLAGS.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time, loss


def sample():
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( FLAGS.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  if FLAGS.use_sh:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )
  print( "done reading and normalizing data." )

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = 128
    model = create_model(sess, actions, batch_size)
    print("Model loaded")

    for key2d in test_set_2d.keys():

      (subj, b, fname) = key2d
      print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if FLAGS.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and FLAGS.camera_frame else key3d

      enc_in  = test_set_2d[ key2d ]
      n2d, _ = enc_in.shape
      dec_out = test_set_3d[ key3d ]
      n3d, _ = dec_out.shape
      assert n2d == n3d

      # Split into about-same-size batches
      enc_in   = np.array_split( enc_in,  n2d // batch_size )
      dec_out  = np.array_split( dec_out, n3d // batch_size )
      all_poses_3d = []

      for bidx in range( len(enc_in) ):

        # Dropout probability 0 (keep probability 1) for sampling
        dp = 1.0
        _, _, poses3d = model.step(sess, enc_in[bidx], dec_out[bidx], dp, isTraining=False)

        # denormalize
        enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
        dec_out[bidx] = data_utils.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d )
        poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
        all_poses_3d.append( poses3d )

      # Put all the poses together
      enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, all_poses_3d] )

      # Convert back to world coordinates
      if FLAGS.camera_frame:
        N_CAMERAS = 4
        N_JOINTS_H36M = 32

        # Add global position back
        dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1,N_JOINTS_H36M] )

        # Load the appropriate camera
        subj, _, sname = key3d

        cname = sname.split('.')[1] # <-- camera name
        scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
        scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
        the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
        R, T, f, c, k, p, name = the_cam
        assert name == cname

        def cam2world_centered(data_3d_camframe):
          data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
          data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
          # subtract root translation
          return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

        # Apply inverse rotation and translation
        dec_out = cam2world_centered(dec_out)
        poses3d = cam2world_centered(poses3d)

  # Grab a random batch to visualize
  enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
  idx = np.random.permutation( enc_in.shape[0] )
  enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]

  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p	= 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx, exidx = 1, 1
  nsamples = 15
  for i in np.arange( nsamples ):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = enc_in[exidx,:]
    viz.show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d gt
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = dec_out[exidx,:]
    viz.show3Dpose( p3d, ax2 )

    # Plot 3d predictions
    ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
    p3d = poses3d[exidx,:]
    viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

    exidx = exidx + 1
    subplot_idx = subplot_idx + 3

  plt.show()


def hankgogo( gogodata,gogodatafake ):
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( FLAGS.action )

  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14 )

  #if FLAGS.use_sh:
  #  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)
  #else:
  train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, FLAGS.data_dir, rcams )
  print( "done reading and normalizing data." )

  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.Session(config=tf.ConfigProto( device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = 1
    model = create_model_my(sess, actions, batch_size)
    print("Model loaded")

    # Dropout probability 0 (keep probability 1) for sampling
    dp = 1.0
    poses3d = model.step(sess, gogodata, isTraining=False)
    tesmp = poses3d
    poses3d = data_utils.unNormalizeData( poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    model.saver.save(sess, os.path.join(mysave_dir,"gogo"))
    

  # Grab a random batch to visualize
 # enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
 # idx = np.random.permutation( enc_in.shape[0] )
 # enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]
  
  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p	= 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx, exidx = 1, 1
  nsamples = 1
  # Plot 2d pose
  #ax1 = plt.subplot(gs1[subplot_idx-1])
  #p2d = enc_in[exidx,:]
  #viz.show2Dpose( p2d, ax1 )
  #ax1.invert_yaxis()

  # Plot 3d gt
  #ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
  #p3d = dec_out[exidx,:]
  #viz.show3Dpose( p3d, ax2 )

  # Plot 3d predictions
  ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
  p3d = poses3d
  viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

  exidx = exidx + 1
  subplot_idx = subplot_idx + 3

  plt.show()


def main(_):
  if FLAGS.hanksam:
    gogodata = np.zeros((1, 32), dtype=np.float64)
   
    gogofake = np.zeros((1, 48), dtype=np.float32)
    gogodata[0:] = [-1.743782355560375,-0.33879302250670057,-1.5613714887709973,-0.3343190142548892,-1.4785489979205242,-0.2078166269987009
    ,-1.3833748088337143,0.1446066964969405,-1.8879590990898538,-0.2239423843222626,-1.7977740289166975,-0.0515661176226813
    ,-1.7560305471433284,0.26848424176777697,-1.6439011159015897,-0.304824180511756,-1.6449827789549991,-0.3869941020547388
    ,-1.7614920302843315,-0.4355473258487442,-1.8598847152093743,-0.29211674626818485,-2.0961764412755755,-0.24036621994164523
    ,-2.2454260783345967,-0.283080128485318,-1.2869616740279937,-0.4059530275702949,-1.5105229517228573,-0.4035138421651531,-1.794821761159529
    ,-0.3321873147555136]

    datamean = np.zeros((1, 32), dtype=np.float64)
    datamean = np.array([534.2362067815909,425.88147731177776,533.393731182906,425.6983578158579,534.3978023429365,497.43802989193154,532.481293821069
    ,569.0466234375722,535.2915655822208,425.73629461606714,532.7675617662295,496.4731547066457,530.8880341233735,569.6075068344198,535.7534460606558
    ,331.2267732306162,536.3380005282891,317.4499285783894,536.7162946417122,269.119694669409,536.3674026383682,330.27798906492825,535.8566970903066
    ,374.59944401417664,534.7028848175864,387.35266055116455,534.9920256553606,331.7734652688376,534.611308079746,373.81679138734876,535.2152919182024
    ,379.5077967523042])

    datastddev = np.zeros((1, 32), dtype=np.float64)
    datastddev = np.array([107.37361012086936,61.634909589570334,114.89497052627252,61.911997024725586,117.27565510971122,50.22711629323473,118.89857526011964
    ,55.0000570865409,102.91089763326161,61.338520877316505,106.6694471505881,47.96084756161423,108.13481259325421,53.60647265856601
    ,110.5622742770474,72.91669969652871,110.84492972268202,76.09916642663414,115.08215260502243,82.92943734420392,105.04274863959776
    ,72.84070268738881,106.31581039747435,73.21929020828857,108.0876752788032,82.49487760274116,124.31763034139937,73.34214366385531,132.80917569040227
    ,76.37108859015507,131.60933137204532,88.82878858279597])
   
    
    gogodata2 =  np.array([224.39846801757812,	217.98716735839844,
    229.39846801757812,	226.98716735839844,
    187.4209442138672, 285.7557373046875,	
    137.04791259765625,	285.7557373046875,
    257.38348388671875,	221.3901824951172,	
    302.1595153808594,	218.59165954589844,	
    338.54010009765625,	277.3602294921875,
    192.22694396972656,	149.24861145019531,
    167.83143615722656,	134.63662719726562,
    176.22694396972656,	120.24861145019531,	
    207.0104522705078,	134.63662719726562,
    221.0029754638672, 95.45761108398438,	
    226.59996032714844,59.07709884643555,
    176.22694396972656,	165.420166015625,	
    128.65240478515625,	157.024658203125,	
    97.86888885498047,	134.63662719726562,])

    gogodatalike =  np.array([-1.6506496017232,
1.9650961361793,
-1.770257916872,
1.9592590776178,
-1.7428834838034,
3.117080606301,
-1.7450275864715,
3.3809669737222,
-1.4701219118831,
1.9769584210626,
-1.3759100256612,
3.2844883546101,
-1.2843970483936,
3.3837796849817,
-1.6077224100441,
1.0391752106821,
-1.5096585919351,
0.71682876413898,
-1.4486720214029,
0.38442688569404,
-1.1458896896477,
0.92972758961037,
-0.52538467121193,
0.29228029833319,
0.17853205865181,
-0.11337262170632,
-1.7374207106603,
0.88934589954326,
-2.1505389713851,
0.25118417148141,
-2.7370041938737,
-0.039489413379027,
])

    gogodata3 =  np.array([ 143.05165100097656,	273.41607666015625,
      131.05165100097656,	284.41607666015625,
    140.84312438964844,	341.2066345214844,
    160.4260711669922,	392.122314453125,
    164.34266662597656,	280.49951171875,
    172.1758270263672,	337.2900695800781,
    189.8004913330078,	390.16400146484375,
    131.13504791259766,208.37620544433594,	
    127.13504791259766,192.37620544433594,	
    126.13504791259766,197.37620544433594,	
    160.4260711669922,	210.00086975097656,	
    193.7170867919922,	229.5838165283203,
    187.8422088623047,	262.8748474121094,
    109.5103988647461,	217.83404541015625,
    80.13597106933594,	245.25018310546875,
    105.59381103515625,	274.6246337890625,])

    gogonormal = np.array([-1.743782355560375,-0.33879302250670057,-1.5613714887709973,-0.3343190142548892,-1.4785489979205242,-0.2078166269987009
    ,-1.3833748088337143,0.1446066964969405,-1.8879590990898538,-0.2239423843222626,-1.7977740289166975,-0.0515661176226813
    ,-1.7560305471433284,0.26848424176777697,-1.6439011159015897,-0.304824180511756,-1.6449827789549991,-0.3869941020547388
    ,-1.7614920302843315,-0.4355473258487442,-1.8598847152093743,-0.29211674626818485,-2.0961764412755755,-0.24036621994164523
    ,-2.2454260783345967,-0.283080128485318,-1.2869616740279937,-0.4059530275702949,-1.5105229517228573,-0.4035138421651531,-1.794821761159529
    ,-0.3321873147555136])

    gogofinal = np.multiply(gogonormal,datastddev) +datamean
    idex = 0
    total = 32
    while idex<total :
      if idex%2==0 :
        gogodata2[idex] = gogodata2[idex] +150
      else:
          gogodata2[idex] = gogodata2[idex] +330
      idex = idex+1
    
  
    gogodatatrue = np.zeros((1, 32), dtype=np.float64)
    #gogodatatrue[0:] = np.divide( (gogodata3 - datamean), datastddev )


    #sample()
    gogodatatrue[0:] = gogodatalike
    #hankgogo(gogodatalike,gogofake)
    hankgogo(gogodatatrue,gogofake)

  else:
    train()

if __name__ == "__main__":
  tf.app.run()
