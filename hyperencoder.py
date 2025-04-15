# hyperencoder.py (with integrated HyperpriorSC using TFC)

# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp # Needed if hyperprior uses TFP distributions directly (optional)

# === ADD TFC Import (Needed because HyperpriorSC uses it) ===
try:
    import tensorflow_compression as tfc
except ImportError:
    print("TensorFlow Compression not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-compression"])
    import tensorflow_compression as tfc
    print("TensorFlow Compression installed successfully.")
# === END TFC Import ===


# Make sure tensorflow-addons is installed: pip install tensorflow-addons
try:
    import tensorflow_addons as tfa
except ImportError:
    print("TensorFlow Addons not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-addons"])
    import tensorflow_addons as tfa
    print("TensorFlow Addons installed successfully.")

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
from glob import glob
from tqdm import tqdm
import cv2 # Make sure opencv-python is installed: pip install opencv-python
import datetime
import argparse # For command-line arguments
import sys # For tf.print redirection
import pickle # To save/load compressed representations

# --- REMOVE Hyperprior Import ---
# try:
#     from hyperprior import HyperpriorSC # Use the advanced version
# except ImportError:
#     print("ERROR: Could not import HyperpriorSC from hyperprior.py.")
#     print("Please ensure hyperprior.py exists and contains the HyperpriorSC class.")
#     sys.exit(1)

# --- Configuration ---
# RAFT specific
IMG_HEIGHT = 384
IMG_WIDTH = 512
NUM_ITERATIONS = 8 # RAFT iterations
RAFT_CHECKPOINT_PATH = './raft_checkpoints/raft_final_weights.weights.h5' # ADJUST PATH AS NEEDED

# Autoencoder specific
AE_BATCH_SIZE = 2
AE_EPOCHS = 200
AE_LEARNING_RATE = 1e-4
AE_CHECKPOINT_DIR = './flow_ae_vq_tfc_checkpoints' # Directory for this VQ+TFC version
AE_LOG_DIR = './flow_ae_vq_tfc_logs'          # Log directory for this VQ+TFC version
LATENT_DIM = 64        # Main latent channels (encoder output)
NUM_EMBEDDINGS = 512     # VQ codebook size
COMMITMENT_COST = 0.25 # VQ commitment loss weight

# HyperpriorSC specific (Match defaults or override via args)
HYPERPRIOR_LATENT_DIM = 32
HYPERPRIOR_FILTERS = 96
CONTEXT_FILTERS = 96
JOINT_FILTERS = 128
# Rate-Distortion trade-off parameter (Higher beta -> more compression focus)
# Can be overridden by command-line argument --beta
DEFAULT_BETA_RATE_DISTORTION = 0.05

# Shared / General
DATA_DIR = './video_data' # Base directory containing scene folders (imX.png)
FLOW_OUTPUT_DIR = './flow_data_generated' # Directory to store/load pre-computed flow maps
# Visualization settings
VISUALIZATION_SCENE = '00081' # Example scene name within DATA_DIR
VISUALIZATION_SEQUENCE = '0001'     # Example sequence ID within the scene
# Output for compressed representations
COMPRESSED_OUTPUT_DIR = './compressed_output_vq_tfc' # Use a specific dir name

# Ensure directories exist
os.makedirs(AE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AE_LOG_DIR, exist_ok=True)
os.makedirs(FLOW_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPRESSED_OUTPUT_DIR, exist_ok=True) # Ensure compressed output dir exists

# --- Prevent TensorFlow from allocating all GPU memory ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

# --- Mixed Precision ---
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16 precision.")
except Exception as e:
    print(f"Could not set mixed precision: {e}. Using default float32.")
    tf.keras.mixed_precision.set_global_policy('float32')


# --- 1. RAFT Model Implementation ---
# [ === Full RAFT Model Code === ] (Assumed to be correct and complete)
# --- Start of RAFT Model Definitions ---
def BasicBlock(filters, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Keep Norm in float32
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Keep Norm in float32
    ], name=f'BasicBlock_f{filters}_s{stride}')

def DownsampleBlock(filters, stride=2):
     return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Keep Norm in float32
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Keep Norm in float32
        tf.keras.layers.ReLU(),
    ], name=f'DownsampleBlock_f{filters}_s{stride}')

class FeatureEncoder(tf.keras.Model):
    def __init__(self, name='feature_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32) # Keep Norm in float32
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = BasicBlock(64, stride=1) # Note: stride=1 in original RAFT basic block
        self.layer2 = DownsampleBlock(96, stride=2)
        self.layer3 = DownsampleBlock(128, stride=2)
        self.conv_out = tf.keras.layers.Conv2D(256, kernel_size=1) # Final 1x1 conv
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x) # Output features at 1/8 resolution
        return x

class ContextEncoder(tf.keras.Model):
    def __init__(self, name='context_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32) # Keep Norm in float32
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = BasicBlock(64, stride=1)
        self.layer2 = DownsampleBlock(96, stride=2)
        self.layer3 = DownsampleBlock(128, stride=2)
        # Output dim for context: 128 (matches hidden_dim + inp_dim usually)
        self.conv_out = tf.keras.layers.Conv2D(128, kernel_size=1)
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x) # Output context features at 1/8 resolution
        return x

class ConvGRUCell(tf.keras.layers.Layer):
    # Standard ConvGRU Cell implementation
    def __init__(self, hidden_filters, input_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters
        self.input_filters = input_filters # Not strictly needed if input shape is known
        self.kernel_size = kernel_size
        self.state_size = tf.TensorShape([None, None, hidden_filters]) # H, W, C

        # Update gate convolutions
        self.conv_update_w = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_update_w')
        self.conv_update_u = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_update_u')
        # Reset gate convolutions
        self.conv_reset_w = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_reset_w')
        self.conv_reset_u = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_reset_u')
        # Candidate state convolutions
        self.conv_candidate_w = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_candidate_w')
        self.conv_candidate_u = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', kernel_initializer='glorot_uniform', name='gru_candidate_u')

    def build(self, input_shape):
        # Build the internal conv layers
        # Pass expected input shape to ensure weights are created
        gru_input_shape = input_shape
        hidden_state_shape = tf.TensorShape(input_shape[:-1].as_list() + [self.hidden_filters])

        self.conv_update_w.build(gru_input_shape)
        self.conv_update_u.build(hidden_state_shape)
        self.conv_reset_w.build(gru_input_shape)
        self.conv_reset_u.build(hidden_state_shape)
        self.conv_candidate_w.build(gru_input_shape)
        self.conv_candidate_u.build(hidden_state_shape)
        self.built = True


    def call(self, inputs, states):
        # inputs: current input tensor (e.g., motion features + context 'inp')
        # states: list containing previous hidden state [h_prev]
        h_prev = states[0]

        # Calculate update gate (z_t)
        update_gate = tf.sigmoid(self.conv_update_w(inputs) + self.conv_update_u(h_prev))

        # Calculate reset gate (r_t)
        reset_gate = tf.sigmoid(self.conv_reset_w(inputs) + self.conv_reset_u(h_prev))

        # Calculate candidate hidden state (h_tilde_t)
        candidate_h = tf.tanh(self.conv_candidate_w(inputs) + self.conv_candidate_u(reset_gate * h_prev))

        # Calculate new hidden state (h_t)
        new_h = (1. - update_gate) * h_prev + update_gate * candidate_h

        # Return new state (as value) and list of new states (for RNN layer)
        return new_h, [new_h]

class UpdateBlock(tf.keras.Model):
    def __init__(self, iterations, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='update_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iterations = iterations
        self.hidden_dim = hidden_dim # Dimension of GRU hidden state 'net'

        # Calculate total channels for correlation features
        corr_feature_dim = (2 * corr_radius + 1)**2 * corr_levels
        # Output dimension of the small motion encoder network
        motion_encoder_output_dim = 128 # Standard RAFT uses 128 here

        # Input dim 'inp' from context encoder split
        inp_dim = context_dim - hidden_dim # context_dim should be sum of hidden + inp
        if inp_dim < 0:
            raise ValueError(f"UpdateBlock init: context_dim ({context_dim}) must be >= hidden_dim ({hidden_dim})")

        # Calculate total GRU input dim based on what's concatenated inside call()
        # gru_input = tf.concat([motion_features, inp], axis=-1)
        # motion_features comes from motion_encoder, inp comes from context split
        gru_input_total_dim = motion_encoder_output_dim + inp_dim


        # Motion Encoder: Takes correlation + current flow estimate, outputs motion features
        # Input channels: corr_feature_dim + 2 (flow dx, dy)
        self.motion_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, corr_feature_dim + 2)), # Explicit input shape can help
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 1, padding='same', activation='relu'), # Project combined input
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 3, padding='same', activation=None) # Output motion features (linear)
        ], name='motion_encoder')

        # GRU Cell: Takes combined features (motion_features + inp), updates hidden state
        self.gru_cell = ConvGRUCell(hidden_filters=hidden_dim, input_filters=gru_input_total_dim) # Pass total expected input dims

        # Flow Head: Takes GRU hidden state, predicts flow delta
        self.flow_head = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, None, hidden_dim)), # Explicit input shape
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2, 3, padding='same') # Predicts dx, dy (linear)
        ], name='flow_head')

    def call(self, net, inp, corr_features, flow_init=None):
        # net: Initial hidden state from context encoder (h_0)
        # inp: Input features from context encoder (c_t) - constant through iterations
        # corr_features: Correlation volume features (lookup based on current flow)
        # flow_init: Optional initial flow estimate (usually zeros)

        shape_tensor = tf.shape(net) # Get shape dynamically
        b, h, w = shape_tensor[0], shape_tensor[1], shape_tensor[2]

        # Initialize flow estimate if not provided
        flow = tf.zeros([b, h, w, 2], dtype=net.dtype) if flow_init is None else tf.cast(flow_init, net.dtype)

        hidden_state = net # Initial hidden state
        flow_predictions = []

        for _ in range(self.iterations):
            # Stop gradient flow through the flow estimate (treat as fixed for this iteration's input)
            flow = tf.stop_gradient(flow)

            # 1. Index correlation features using the current flow estimate
            #    In a full RAFT, this involves a lookup function based on 'flow'.
            #    Here, assuming `corr_features` is the volume based on zero flow,
            #    or lookup is handled externally. Let's assume it's ready.

            # 2. Motion Encoder Input: Concatenate correlation and current flow
            motion_input = tf.concat([corr_features, flow], axis=-1)
            motion_features = self.motion_encoder(motion_input)

            # 3. GRU Input: Concatenate motion features and context 'inp'
            gru_input = tf.concat([motion_features, inp], axis=-1)

            # 4. Update GRU hidden state
            hidden_state, [hidden_state] = self.gru_cell(gru_input, [hidden_state])

            # 5. Predict flow delta using Flow Head
            delta_flow = self.flow_head(hidden_state)
            # Ensure delta_flow is float32 for accumulation
            delta_flow = tf.cast(delta_flow, tf.float32)


            # 6. Update flow estimate
            # Ensure base flow is also float32 for accumulation
            flow = tf.cast(flow, tf.float32) + delta_flow
            flow_predictions.append(flow) # Store prediction for this iteration

        return flow_predictions # Return list of flow estimates from each iteration

def build_correlation_volume(fmap1, fmap2, radius=4):
    """ Computes all pairwise correlations (dot products) between fmap1 and
        neighborhoods in fmap2. Optimized implementation needed for performance.
        This is a basic, understandable version.
    """
    compute_dtype = fmap1.dtype # Should match policy.compute_dtype
    fmap2 = tf.cast(fmap2, compute_dtype) # Ensure same compute dtype

    batch_size, h, w, c = tf.shape(fmap1)[0], tf.shape(fmap1)[1], tf.shape(fmap1)[2], tf.shape(fmap1)[3]
    pad_size = radius
    # Pad fmap2 for neighborhood lookups
    fmap2_padded = tf.pad(fmap2, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')

    # List to store correlation features for each displacement
    correlation_patches = []
    # Iterate through all displacements in the neighborhood
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Shift fmap2_padded to align neighborhoods
            fmap2_shifted = tf.slice(fmap2_padded,
                                     [0, pad_size + dy, pad_size + dx, 0],
                                     [batch_size, h, w, c])
            # Compute dot product (correlation) for this shift
            correlation = tf.reduce_sum(fmap1 * fmap2_shifted, axis=-1, keepdims=True)
            correlation_patches.append(correlation)

    # Concatenate all correlation patches along the channel dimension
    correlation_volume = tf.concat(correlation_patches, axis=-1)

    # Normalize (optional but common) - divide by number of channels 'c'
    # Cast 'c' safely
    c_float = tf.cast(c, correlation_volume.dtype)
    correlation_normalized = correlation_volume / tf.maximum(c_float, 1e-6)

    # Return as float32 for subsequent calculations (e.g., GRU)
    return tf.cast(correlation_normalized, tf.float32)

class RAFT(tf.keras.Model):
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim       # GRU hidden state dim
        self.context_dim = context_dim     # Context encoder output dim (hidden_dim + inp_dim)
        self.corr_levels = corr_levels     # Number of correlation pyramid levels (not used in this basic build)
        self.corr_radius = corr_radius     # Radius for correlation neighborhood

        # Encoders
        self.feature_encoder = FeatureEncoder() # Encodes both images
        self.context_encoder = ContextEncoder() # Encodes only image1

        # Update Block
        self.update_block = UpdateBlock(iterations=num_iterations, hidden_dim=hidden_dim, context_dim=context_dim, corr_levels=corr_levels, corr_radius=corr_radius)

    @tf.function # Decorate for performance
    def upsample_flow(self, flow, target_height, target_width):
        """Upsamples flow field to target resolution using bilinear interpolation."""
        # Input flow is at 1/8 resolution, target is full resolution
        flow = tf.cast(flow, tf.float32) # Ensure float32 for resize and scaling
        shape_tensor = tf.shape(flow)
        b, h_low, w_low = shape_tensor[0], shape_tensor[1], shape_tensor[2]

        # Calculate scaling factors (target_dim / low_res_dim)
        # Avoid division by zero if dimensions are somehow 0
        h_low_safe = tf.maximum(tf.cast(h_low, tf.float32), 1.0)
        w_low_safe = tf.maximum(tf.cast(w_low, tf.float32), 1.0)
        scale_factor_h = tf.cast(target_height, tf.float32) / h_low_safe
        scale_factor_w = tf.cast(target_width, tf.float32) / w_low_safe

        # Upsample using bilinear interpolation
        flow_upsampled = tf.image.resize(flow, [target_height, target_width], method='bilinear')

        # Scale the flow values
        u = flow_upsampled[..., 0] * scale_factor_w
        v = flow_upsampled[..., 1] * scale_factor_h
        flow_scaled = tf.stack([u, v], axis=-1)
        return flow_scaled

    def call(self, inputs, training=False):
        image1, image2 = inputs # Expect a tuple/list of two images
        target_height = tf.shape(image1)[1]
        target_width = tf.shape(image1)[2]

        # 1. Feature Extraction (Shared weights for both images)
        # Outputs are at 1/8 resolution
        fmap1 = self.feature_encoder(image1)
        fmap2 = self.feature_encoder(image2)

        # 2. Context Extraction (Only from image1)
        # Output is at 1/8 resolution
        context_fmap = self.context_encoder(image1)

        # 3. Split Context into GRU hidden state 'net' and input 'inp'
        # Ensure split sizes match context_dim
        inp_dim = self.context_dim - self.hidden_dim
        if inp_dim < 0:
             raise ValueError(f"Context split results in negative dimension for 'inp': hidden={self.hidden_dim}, context={self.context_dim}")
        split_sizes = [self.hidden_dim, inp_dim]


        net, inp = tf.split(context_fmap, split_sizes, axis=-1)
        # Apply activations as in original RAFT
        net = tf.tanh(net)
        inp = tf.nn.relu(inp)

        # 4. Build Correlation Volume (using 1/8 resolution features)
        corr_features = build_correlation_volume(fmap1, fmap2, radius=self.corr_radius)

        # 5. Iterative Updates
        flow_predictions_low_res = self.update_block(net, inp, corr_features, flow_init=None)

        # 6. Upsample all intermediate flow predictions to full resolution
        flow_predictions_upsampled = [self.upsample_flow(flow_lr, target_height, target_width)
                                      for flow_lr in flow_predictions_low_res]

        # Return the list of upsampled flow predictions (last one is the final estimate)
        return flow_predictions_upsampled
# --- End of RAFT Model Definitions ---


# --- 2. Dataset Loading and Flow Generation/Loading Functions ---
# --- Start Dataset Functions ---
def load_and_preprocess_image(path):
    """Loads and preprocesses a single image."""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
        img = tf.image.convert_image_dtype(img, tf.float32) # Scale to [0, 1]
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return img
    except Exception as e:
        tf.print(f"Error loading image {path}: {e}", output_stream=sys.stderr)
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

def find_image_pairs(data_dir):
    """Finds consecutive image pairs (imX.png, imX+1.png) in the dataset structure."""
    def parse_frame_num(filename):
        basename = os.path.basename(filename)
        match = re.search(r'im(\d+)\.(png|jpg|jpeg|bmp|tiff)$', basename, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    image_pairs = []
    print(f"Scanning dataset structure for image pairs in: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return image_pairs

    scene_folders = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()])
    if not scene_folders:
        print(f"Warning: No numeric scene folders found directly in {data_dir}.")

    for scene_path in tqdm(scene_folders, desc="Scanning Scenes"):
        sequence_folders = sorted([os.path.join(scene_path, s) for s in os.listdir(scene_path)
                                   if os.path.isdir(os.path.join(scene_path, s)) and s.isdigit()])
        for seq_path in sequence_folders:
            frame_pattern = os.path.join(seq_path, 'im*.*') # Look for common image extensions
            frame_files = glob(frame_pattern)
            try:
                valid_frames = [(f, parse_frame_num(f)) for f in frame_files
                                if parse_frame_num(f) != -1]
                if len(valid_frames) < 2: continue # Need at least two frames for a pair
                sorted_frames_with_nums = sorted(valid_frames, key=lambda item: item[1])
                sorted_frames = [item[0] for item in sorted_frames_with_nums] # Just paths
            except Exception as e:
                 print(f"\nWarning: Error processing frames in {seq_path}: {e}. Skipping.", file=sys.stderr)
                 continue

            for i in range(len(sorted_frames) - 1):
                frame1_path = sorted_frames[i]
                frame2_path = sorted_frames[i+1]
                frame1_num = sorted_frames_with_nums[i][1]
                frame2_num = sorted_frames_with_nums[i+1][1]

                if frame2_num == frame1_num + 1:
                    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
                        image_pairs.append((frame1_path, frame2_path))
                    else:
                         print(f"\nWarning: Missing frame file detected after sorting for pair ({frame1_path}, {frame2_path}). Skipping.", file=sys.stderr)


    print(f"Found {len(image_pairs)} consecutive image pairs.")
    return image_pairs


def get_flow_output_path(frame1_path, data_dir, flow_output_dir):
    """Determines the output path for the flow map corresponding to frame1_path."""
    try:
        rel_path = os.path.relpath(frame1_path, data_dir)
        rel_dir, filename = os.path.split(rel_path)
        base, _ = os.path.splitext(filename)
        flow_filename = f"flow_{base}.npy"
        output_path = os.path.join(flow_output_dir, rel_dir, flow_filename)
        return output_path
    except ValueError as e:
        print(f"Error determining flow output path for {frame1_path} relative to {data_dir}: {e}", file=sys.stderr)
        return None

def generate_flow_maps(data_dir, flow_output_dir, raft_model, batch_size=4, force_regenerate=False):
    """Generates flow maps using RAFT and saves them to flow_output_dir."""
    print("\n--- Starting Flow Generation Phase ---")
    image_pairs = find_image_pairs(data_dir)
    if not image_pairs:
        print("No image pairs found. Skipping flow generation.")
        return

    generated_count = 0; skipped_count = 0; error_count = 0
    img1_paths = [p[0] for p in image_pairs]
    img2_paths = [p[1] for p in image_pairs]
    output_flow_paths = [get_flow_output_path(p[0], data_dir, flow_output_dir) for p in image_pairs]

    valid_indices = [i for i, p in enumerate(output_flow_paths) if p is not None]
    if len(valid_indices) != len(image_pairs):
        print(f"Warning: {len(image_pairs) - len(valid_indices)} pairs skipped due to output path errors.", file=sys.stderr)
        img1_paths = [img1_paths[i] for i in valid_indices]
        img2_paths = [img2_paths[i] for i in valid_indices]
        output_flow_paths = [output_flow_paths[i] for i in valid_indices]
        if not img1_paths:
            print("No valid pairs remaining after output path filtering. Skipping flow generation.")
            return

    path_ds = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths))
    image_ds = path_ds.map(lambda p1, p2: (load_and_preprocess_image(p1), load_and_preprocess_image(p2)),
                           num_parallel_calls=tf.data.AUTOTUNE)
    batched_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    pair_idx = 0
    total_pairs = len(img1_paths)
    total_batches = tf.data.experimental.cardinality(batched_ds).numpy()
    if total_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        total_batches = (total_pairs + batch_size - 1) // batch_size # Estimate

    print(f"Processing {total_pairs} pairs in approximately {total_batches} batches...")

    for img1_batch_tf, img2_batch_tf in tqdm(batched_ds, desc="Generating Flow Maps", total=total_batches):
        batch_start_idx = pair_idx
        current_batch_actual_size = tf.shape(img1_batch_tf)[0].numpy()
        try:
            if not raft_model.built:
                _ = raft_model([img1_batch_tf, img2_batch_tf], training=False)
                print("\nBuilt RAFT model during first batch processing.")

            flow_predictions_list = raft_model([img1_batch_tf, img2_batch_tf], training=False)
            final_flow_batch_tf = flow_predictions_list[-1]
            final_flow_batch_np = final_flow_batch_tf.numpy()

            for i in range(current_batch_actual_size):
                if pair_idx >= total_pairs:
                    print(f"\nWarning: pair_idx ({pair_idx}) exceeded total_pairs ({total_pairs}). Stopping batch processing.", file=sys.stderr)
                    break
                output_path = output_flow_paths[pair_idx]
                if not force_regenerate and os.path.exists(output_path):
                    skipped_count += 1
                    pair_idx += 1
                    continue
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, final_flow_batch_np[i].astype(np.float32))
                generated_count += 1
                pair_idx += 1
        except Exception as e:
            print(f"\nError during flow generation for batch starting at original index {batch_start_idx}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            error_count += current_batch_actual_size
            pair_idx = batch_start_idx + current_batch_actual_size
            print(f"Skipping potentially failed batch. Advanced index to {pair_idx}.", file=sys.stderr)

    print("\n--- Flow Generation Summary ---")
    print(f"Flow maps generated: {generated_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors encountered (estimated pairs affected): {error_count}")
    print(f"Total pairs attempted: {total_pairs}")
    print(f"Flow maps saved in: {flow_output_dir}")
    print("-" * 30)


def load_flow_map(flow_path_tensor):
    """Loads a .npy flow map using tf.py_function."""
    def _load_npy(path_tensor):
        path_bytes = path_tensor.numpy()
        path_str = path_bytes.decode('utf-8')
        try:
            flow_map = np.load(path_str)
            if flow_map.shape != (IMG_HEIGHT, IMG_WIDTH, 2):
                 tf.print(f"Warning: Flow map {path_str} shape {flow_map.shape} != expected {(IMG_HEIGHT, IMG_WIDTH, 2)}. Resizing.", output_stream=sys.stderr)
                 flow_map_resized = cv2.resize(flow_map, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                 if flow_map_resized.ndim == 2:
                      flow_map = np.stack([flow_map_resized, np.zeros_like(flow_map_resized)], axis=-1)
                      tf.print(f"Warning: Flow map {path_str} became 2D after resize. Stacked to {flow_map.shape}.", output_stream=sys.stderr)
                 elif flow_map_resized.shape == (IMG_HEIGHT, IMG_WIDTH, 2):
                     flow_map = flow_map_resized
                 else:
                      tf.print(f"ERROR: Resized flow map {path_str} still has wrong shape {flow_map_resized.shape}. Returning zeros.", output_stream=sys.stderr)
                      return np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float32)
            return flow_map.astype(np.float32)
        except FileNotFoundError:
            tf.print(f"Error loading flow map: File not found at {path_str}", output_stream=sys.stderr)
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float32)
        except Exception as e:
            tf.print(f"Error loading flow map {path_str}: {e}", output_stream=sys.stderr)
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float32)

    flow = tf.py_function(_load_npy, [flow_path_tensor], tf.float32)
    flow.set_shape([IMG_HEIGHT, IMG_WIDTH, 2])
    return flow


def configure_dataset_for_ae_from_flow(flow_dir, batch_size):
    """ Configures dataset for AE training by loading pre-computed flow maps (.npy) """
    print(f"Scanning for pre-computed flow maps (.npy) in: {flow_dir}")
    flow_map_paths = glob(os.path.join(flow_dir, '**', 'flow_im*.npy'), recursive=True)
    if not flow_map_paths:
        raise ValueError(f"No pre-computed flow maps (flow_im*.npy) found in '{flow_dir}' or its subdirectories. Please run with '--mode generate_flow' first or check the directory.")
    print(f"Found {len(flow_map_paths)} potential flow maps.")

    flow_map_paths = [p for p in flow_map_paths if p]
    if not flow_map_paths:
         raise ValueError(f"Found 0 valid flow map paths after filtering in {flow_dir}.")

    flow_path_ds = tf.data.Dataset.from_tensor_slices(flow_map_paths)
    shuffle_buffer_size = max(len(flow_map_paths) // 4, 2 * batch_size)
    print(f"Using shuffle buffer size: {shuffle_buffer_size}")
    flow_path_ds = flow_path_ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    flow_ds = flow_path_ds.map(load_flow_map, num_parallel_calls=tf.data.AUTOTUNE)
    flow_ds = flow_ds.filter(lambda x: tf.reduce_any(tf.not_equal(x, 0.0)))
    flow_ds = flow_ds.batch(batch_size)
    flow_ds = flow_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Dataset configured to load, shuffle, batch, and prefetch pre-computed flow maps.")
    return flow_ds
# --- End Dataset Functions ---

# --------------------------------------------------------------------------
# --- Integrated Hyperprior & Autoencoder Definitions ---
# --------------------------------------------------------------------------

# === MaskedConv2D (Moved from hyperprior.py) ===
class MaskedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, mask_type='A', **kwargs):
        super().__init__(**kwargs)
        if mask_type not in {'A'}:
            raise ValueError(f"Invalid mask type {mask_type}, only 'A' is supported.")
        self.mask_type = mask_type
        self._mask_internal = None

    def build(self, input_shape):
        super().build(input_shape)
        if not self.built: raise ValueError("Build called before kernel created.")
        kernel_shape = tf.shape(self.kernel).numpy()
        h, w = self.kernel_size[0], self.kernel_size[1]
        mid_h, mid_w = h // 2, w // 2
        mask = np.zeros(kernel_shape, dtype=np.float32)
        mask[:mid_h, :, :, :] = 1.0
        mask[mid_h, :mid_w, :, :] = 1.0
        self._mask_internal = tf.constant(mask, dtype=tf.float32)

    def call(self, inputs):
        if self._mask_internal is None: raise RuntimeError("Mask not created.")
        original_kernel_var = self.kernel
        try:
            current_compute_dtype = self.compute_dtype
            mask_casted = tf.cast(self._mask_internal, current_compute_dtype)
            masked_kernel_value = tf.multiply(original_kernel_var, mask_casted)
            self.kernel = masked_kernel_value
            outputs = super().call(inputs)
        finally:
            self.kernel = original_kernel_var
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"mask_type": self.mask_type})
        return config

# === HyperpriorSC (Moved from hyperprior.py, uses TFC) ===
class HyperpriorSC(tf.keras.Model):
    """ Hyperprior Network with Spatial Context (SC) modeling for VQ indices."""
    def __init__(self, num_embeddings, latent_dim, hyper_latent_dim=32,
                 hyper_filters=96, context_filters=96, joint_filters=128,
                 name="hyperprior_sc_tfc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_embeddings = num_embeddings; self.latent_dim = latent_dim
        self.hyper_latent_dim = hyper_latent_dim; self.hyper_filters = hyper_filters
        self.context_filters = context_filters; self.joint_filters = joint_filters
        self.hyper_encoder_model = None; self.hyper_decoder_model = None
        self.context_model = None; self.joint_parameter_predictor = None
        self.context_free_predictor = None
        self.entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
        self.categorical_entropy_model = tfc.CategoricalEntropyModel(
            prior_shape=(num_embeddings,), coding_rank=3, bottleneck_dtype=None)
        self._built = False

    def build(self, main_latent_shape):
        if self._built: return
        main_latent_shape = tf.TensorShape(main_latent_shape)
        print(f"Building HyperpriorSC with main_latent_shape: {main_latent_shape}")
        policy = tf.keras.mixed_precision.global_policy(); compute_dtype = policy.compute_dtype
        # --- 1. Hyper-Encoder ---
        hyper_enc_input=tf.keras.Input(shape=main_latent_shape, name="hyper_enc_input_y"); x=tf.cast(hyper_enc_input, compute_dtype)
        x=tf.keras.layers.Conv2D(self.hyper_filters, 3, 1,'same', activation='relu', name='hyper_enc_conv1')(x)
        x=tf.keras.layers.Conv2D(self.hyper_filters, 5, 2,'same', activation='relu', name='hyper_enc_conv2')(x)
        x=tf.keras.layers.Conv2D(self.hyper_filters, 5, 2,'same', activation='relu', name='hyper_enc_conv3')(x)
        z=tf.keras.layers.Conv2D(self.hyper_latent_dim, 3, 1, 'same', name='hyper_enc_output_z')(x)
        self.hyper_encoder_model=tf.keras.Model(hyper_enc_input, z, name="hyper_encoder"); print(f" HyperpriorSC Encoder built. Output 'z' shape: {z.shape}"); hyper_latent_z_shape=z.shape[1:]
        # --- 2. Hyper-Decoder ---
        hyper_dec_input=tf.keras.Input(shape=hyper_latent_z_shape, name="hyper_dec_input_z"); x=tf.cast(hyper_dec_input, compute_dtype)
        x=tf.keras.layers.Conv2DTranspose(self.hyper_filters, 5, 2, 'same', activation='relu', name='hyper_dec_deconv1')(x)
        x=tf.keras.layers.Conv2DTranspose(self.hyper_filters, 5, 2, 'same', activation='relu', name='hyper_dec_deconv2')(x)
        params_from_z=tf.keras.layers.Conv2D(self.hyper_filters, 3, 1, 'same', name='hyper_dec_params_from_z')(x)
        self.hyper_decoder_model=tf.keras.Model(hyper_dec_input, params_from_z, name="hyper_decoder"); print(f" HyperpriorSC Decoder built. Output 'params_from_z' shape: {params_from_z.shape}"); params_from_z_shape=params_from_z.shape[1:]
        # --- 3. Context Model ---
        ctx_input=tf.keras.Input(shape=main_latent_shape, name="context_input_quantized_y"); x=tf.cast(ctx_input, compute_dtype)
        x=MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', activation='relu', name='ctx_masked_conv1')(x)
        x=MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', activation='relu', name='ctx_masked_conv2')(x)
        context_features=MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', name='ctx_features')(x)
        self.context_model=tf.keras.Model(ctx_input, context_features, name="context_model"); print(f" HyperpriorSC Context Model built. Output 'context_features' shape: {context_features.shape}"); context_features_shape=context_features.shape[1:]
        # --- 4. Joint Parameter Predictor ---
        joint_input_z=tf.keras.Input(shape=params_from_z_shape, name="joint_input_params_z"); joint_input_ctx=tf.keras.Input(shape=context_features_shape, name="joint_input_ctx")
        input_z_casted=tf.cast(joint_input_z, compute_dtype); input_ctx_casted=tf.cast(joint_input_ctx, compute_dtype); joint_features=tf.keras.layers.Concatenate(axis=-1)([input_z_casted, input_ctx_casted])
        x=tf.keras.layers.Conv2D(self.joint_filters, 1, 1,'same', activation='relu', name='joint_conv1')(joint_features)
        x=tf.keras.layers.Conv2D(self.joint_filters, 3, 1,'same', activation='relu', name='joint_conv2')(x)
        logits_context=tf.keras.layers.Conv2D(self.num_embeddings, 1, 1, 'same', activation=None, dtype='float32', name='joint_output_logits_context')(x)
        self.joint_parameter_predictor=tf.keras.Model(inputs=[joint_input_z, joint_input_ctx], outputs=logits_context, name="joint_parameter_predictor"); print(f" HyperpriorSC Joint Predictor built. Output 'logits_context' shape: {logits_context.shape}")
        # --- 5. Context-Free Predictor ---
        cf_input_z=tf.keras.Input(shape=params_from_z_shape, name="cf_input_params_z"); cf_input_z_casted=tf.cast(cf_input_z, compute_dtype)
        cf_x=tf.keras.layers.Conv2D(self.joint_filters, 1, 1, 'same', activation='relu', name='cf_conv1')(cf_input_z_casted)
        logits_context_free=tf.keras.layers.Conv2D(self.num_embeddings, 1, 1, 'same', activation=None, dtype='float32', name='cf_output_logits')(cf_x)
        self.context_free_predictor=tf.keras.Model(inputs=cf_input_z, outputs=logits_context_free, name="context_free_predictor"); print(f" HyperpriorSC Context-Free Predictor built. Output 'logits_context_free' shape: {logits_context_free.shape}")
        self._built = True

    @tf.function
    def get_logits_with_context(self, y, quantized_y, training=False):
        """Gets logits using both hyperprior and context model (for training rate)."""
        if not self._built: self.build(tf.shape(y)[1:])
        y_input_he = tf.stop_gradient(y) if training else y
        z = self.hyper_encoder_model(y_input_he, training=training)
        z_hat, _ = self.entropy_bottleneck(tf.cast(z, tf.float32), training=False)
        z_hat_compute = tf.cast(z_hat, y.dtype)
        params_from_z = self.hyper_decoder_model(z_hat_compute, training=training)
        context_features = self.context_model(quantized_y, training=training)
        policy = tf.keras.mixed_precision.global_policy()
        params_from_z_casted = tf.cast(params_from_z, policy.compute_dtype)
        context_features_casted = tf.cast(context_features, policy.compute_dtype)
        final_logits = self.joint_parameter_predictor([params_from_z_casted, context_features_casted], training=training)
        return final_logits

    @tf.function
    def get_logits_context_free(self, z_hat):
        """Gets logits using only z_hat (for context-free coding)."""
        if not self._built: raise RuntimeError("Model must be built first.")
        z_hat_compute = tf.cast(z_hat, tf.keras.mixed_precision.global_policy().compute_dtype)
        params_from_z = self.hyper_decoder_model(z_hat_compute, training=False)
        logits = self.context_free_predictor(params_from_z, training=False)
        return tf.cast(logits, tf.float32)

    @tf.function
    def call(self, inputs, training):
        """ Forward pass for training rate calculation."""
        if not isinstance(inputs, tuple) or len(inputs) != 3: raise ValueError("Expects (y, quantized_y, indices)")
        y, quantized_y, indices = inputs
        logits = self.get_logits_with_context(y, quantized_y, training=training)
        indices_int = tf.cast(indices, tf.int32); logits_f32 = tf.cast(logits, tf.float32)
        _, bits = self.categorical_entropy_model(indices_int, logits_f32, training=training)
        num_elements = tf.cast(tf.reduce_prod(tf.shape(indices_int)), tf.float32)
        if num_elements == 0: return tf.constant(0.0, dtype=tf.float32)
        rate_loss_bits_per_element = tf.reduce_sum(bits) / num_elements
        return rate_loss_bits_per_element

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), # y
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int64) # indices
    ])
    def compress(self, y, indices):
        """ Compresses y (via z) and indices using TFC layers."""
        print("Tracing HyperpriorSC compress..."); y_compute = tf.cast(y, tf.keras.mixed_precision.global_policy().compute_dtype)
        z = self.hyper_encoder_model(y_compute); z_shape = tf.shape(z)
        z_strings = self.entropy_bottleneck.compress(tf.cast(z, tf.float32))
        z_spatial_shape = z_shape[1:-1]; z_hat = self.entropy_bottleneck.decompress(z_strings, z_spatial_shape)
        z_hat = tf.reshape(z_hat, z_shape)
        logits_cf = self.get_logits_context_free(z_hat)
        indices_int = tf.cast(indices, tf.int32)
        indices_strings = self.categorical_entropy_model.compress(indices_int, logits_cf)
        y_shape = tf.shape(y); return [z_strings, indices_strings], [z_shape, y_shape]

    @tf.function(input_signature=[
        [tf.TensorSpec(shape=(None,), dtype=tf.string), tf.TensorSpec(shape=(None,), dtype=tf.string)],
        [tf.TensorSpec(shape=(4,), dtype=tf.int32), tf.TensorSpec(shape=(4,), dtype=tf.int32)]])
    def decompress(self, strings, shapes):
        """ Decompresses strings to indices using TFC layers."""
        print("Tracing HyperpriorSC decompress..."); z_strings, indices_strings = strings; z_shape, _ = shapes
        z_spatial_shape = z_shape[1:-1]; z_hat = self.entropy_bottleneck.decompress(z_strings, z_spatial_shape)
        z_hat = tf.reshape(z_hat, z_shape)
        logits_cf = self.get_logits_context_free(z_hat)
        indices_decoded = self.categorical_entropy_model.decompress(indices_strings, logits_cf)
        indices_decoded = tf.cast(indices_decoded, tf.int64)
        return indices_decoded

# === Vector Quantizer Layer === (Keep as is)
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim; self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost; w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value=w_init(shape=(embedding_dim, num_embeddings), dtype="float32"), trainable=True, name="embeddings_vq")

    def call(self, inputs):
        original_input_dtype = inputs.dtype; inputs_float32 = tf.cast(inputs, dtype=tf.float32)
        flat_inputs = tf.reshape(inputs_float32, [-1, self.embedding_dim])
        distances = ( tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True) - 2 * tf.matmul(flat_inputs, self.embeddings) + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True) )
        encoding_indices = tf.argmin(distances, axis=1) # int64
        quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.embeddings), encoding_indices) # float32
        input_shape = tf.shape(inputs); quantized_spatial = tf.reshape(quantized_flat, input_shape) # float32
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized_spatial) - inputs_float32))
        q_latent_loss = tf.reduce_mean(tf.square(quantized_spatial - tf.stop_gradient(inputs_float32)))
        loss = q_latent_loss + self.commitment_cost * e_latent_loss; self.add_loss(loss)
        ste_calculation_f32 = inputs_float32 + tf.stop_gradient(quantized_spatial - inputs_float32)
        quantized_ste = tf.cast(ste_calculation_f32, original_input_dtype)
        indices_spatial = tf.reshape(encoding_indices, input_shape[:-1]) # int64
        return quantized_ste, indices_spatial

# === Autoencoder Architecture === (Keep as is)
def build_compression_encoder(input_shape, latent_dim, num_filters=96):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32); policy = tf.keras.mixed_precision.global_policy(); x = tf.cast(inputs, policy.compute_dtype); activation_fn = tf.keras.layers.ReLU()
    x = tf.keras.layers.Conv2D(num_filters, 5, 2, 'same', name='enc_conv1')(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2D(num_filters, 5, 2, 'same', name='enc_conv2')(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2D(num_filters, 5, 2, 'same', name='enc_conv3')(x); x = activation_fn(x)
    latent = tf.keras.layers.Conv2D(latent_dim, 3, 1, 'same', name='encoder_output')(x)
    print(f" Encoder Output shape (symbolic): {latent.shape}"); return tf.keras.Model(inputs, latent, name='flow_compression_encoder')

def build_compression_decoder(latent_shape, latent_dim, num_filters=96):
    policy = tf.keras.mixed_precision.global_policy(); latent_inputs = tf.keras.Input(shape=latent_shape, dtype=policy.compute_dtype); x = latent_inputs; activation_fn = tf.keras.layers.ReLU()
    x = tf.keras.layers.Conv2D(num_filters, 3, strides=1, padding='same', name='dec_conv1')(x); x = activation_fn(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='dec_upsample1')(x)
    x = tf.keras.layers.Conv2D(num_filters, 5, strides=1, padding='same', name='dec_conv2')(x); x = activation_fn(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='dec_upsample2')(x)
    x = tf.keras.layers.Conv2D(num_filters, 5, strides=1, padding='same', name='dec_conv3')(x); x = activation_fn(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='dec_upsample3')(x)
    outputs = tf.keras.layers.Conv2D(2, kernel_size=5, strides=1, padding='same', activation='linear', dtype='float32', name='decoder_output')(x)
    print(f" Decoder Output shape (symbolic): {outputs.shape}"); return tf.keras.Model(latent_inputs, outputs, name='flow_compression_decoder')

# === FlowVQAutoencoder Class (using integrated HyperpriorSC) ===
class FlowVQAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, num_embeddings, commitment_cost,
                 beta_rate, num_filters=96, hyper_latent_dim=HYPERPRIOR_LATENT_DIM,
                 hyper_filters=HYPERPRIOR_FILTERS, context_filters=CONTEXT_FILTERS,
                 joint_filters=JOINT_FILTERS, **kwargs):
        super().__init__(name="FlowVQAutoencoder_HyperpriorSC_TFC", **kwargs)
        self.input_flow_shape = input_shape; self.latent_dim = latent_dim; self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost; self.beta_rate = beta_rate; self.num_filters = num_filters

        self.encoder = build_compression_encoder(input_shape, latent_dim, num_filters=self.num_filters)
        dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
        encoded_output = self.encoder(dummy_input)
        self.encoder_output_shape = encoded_output.shape[1:]
        print(f"Encoder output shape determined (H, W, C): {self.encoder_output_shape}")

        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, name="vector_quantizer")
        self.latent_shape = self.encoder_output_shape
        self.decoder = build_compression_decoder(self.latent_shape, latent_dim, num_filters=self.num_filters)

        # Instantiate the *locally defined* HyperpriorSC
        self.hyperprior = HyperpriorSC(
            num_embeddings=num_embeddings, latent_dim=latent_dim,
            hyper_latent_dim=hyper_latent_dim, hyper_filters=hyper_filters,
            context_filters=context_filters, joint_filters=joint_filters)
        self.hyperprior.build(self.encoder_output_shape) # Build hyperprior

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
        self.rate_loss_tracker = tf.keras.metrics.Mean(name="rate_loss_bits")

    @property
    def metrics(self):
         return [self.total_loss_tracker, self.recon_loss_tracker, self.vq_loss_tracker, self.rate_loss_tracker]

    @property
    def compute_dtype(self):
        return tf.keras.mixed_precision.global_policy().compute_dtype

    def call(self, inputs, training=False):
        encoded_y = self.encoder(inputs, training=training)
        quantized_ste, indices = self.vq_layer(encoded_y)
        reconstructed = self.decoder(quantized_ste, training=training)
        return reconstructed, indices, encoded_y, quantized_ste

    @tf.function
    def train_step(self, data):
        flow_target = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            reconstructed_flow, indices, encoded_y, quantized_ste = self(flow_target, training=True)
            recon_loss = reconstruction_loss_fn(tf.cast(flow_target, tf.float32), tf.cast(reconstructed_flow, tf.float32))
            vq_commitment_loss = sum(tf.cast(loss, tf.float32) for loss in self.vq_layer.losses)
            # Rate loss from hyperprior's call method
            rate_loss = self.hyperprior((encoded_y, quantized_ste, indices), training=True)
            total_loss = recon_loss + self.beta_rate * rate_loss + vq_commitment_loss
            optimizer = self.optimizer
            scaled_loss = optimizer.get_scaled_loss(total_loss) if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else total_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
             gradients = optimizer.get_unscaled_gradients(gradients)
        gradients = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in gradients]
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if valid_grads_and_vars: optimizer.apply_gradients(valid_grads_and_vars)
        else: tf.print("Warning: No valid gradients found.", output_stream=sys.stderr)

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_commitment_loss)
        self.rate_loss_tracker.update_state(rate_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        flow_target = data[0] if isinstance(data, tuple) else data
        reconstructed_flow, indices, encoded_y, quantized_ste = self(flow_target, training=False)
        recon_loss = reconstruction_loss_fn(tf.cast(flow_target, tf.float32), tf.cast(reconstructed_flow, tf.float32))
        # Re-calculate VQ loss
        encoded_y_f32 = tf.cast(encoded_y, tf.float32); flat_indices = tf.reshape(indices, [-1])
        quantized_flat_f32 = tf.nn.embedding_lookup(tf.transpose(self.vq_layer.embeddings), flat_indices)
        quantized_spatial_f32 = tf.reshape(quantized_flat_f32, tf.shape(encoded_y_f32))
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized_spatial_f32) - encoded_y_f32))
        q_latent_loss = tf.reduce_mean(tf.square(quantized_spatial_f32 - tf.stop_gradient(encoded_y_f32)))
        vq_commitment_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # Re-calculate rate loss using hyperprior call
        rate_loss = self.hyperprior((encoded_y, quantized_ste, indices), training=False)
        total_loss = recon_loss + self.beta_rate * rate_loss + vq_commitment_loss

        self.total_loss_tracker.update_state(total_loss); self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_commitment_loss); self.rate_loss_tracker.update_state(rate_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 2), dtype=tf.float32)])
    def compress(self, flow_input):
        """ Encodes flow map and uses HyperpriorSC to compress latents. """
        print("Tracing FlowVQAutoencoder compress...")
        y = self.encoder(flow_input, training=False)
        indices = self.encode_to_indices(flow_input) # Get indices [B,Hl,Wl], int64
        strings, shapes = self.hyperprior.compress(tf.cast(y, tf.float32), indices)
        return strings, shapes

    @tf.function(input_signature=[
        [tf.TensorSpec(shape=(None,), dtype=tf.string),
         tf.TensorSpec(shape=(None,), dtype=tf.string)],
        [tf.TensorSpec(shape=(4,), dtype=tf.int32),
         tf.TensorSpec(shape=(4,), dtype=tf.int32)]])
    def decompress(self, strings, shapes):
        """ Decompresses strings using HyperpriorSC and decodes indices. """
        print("Tracing FlowVQAutoencoder decompress...")
        indices_decoded = self.hyperprior.decompress(strings, shapes) # [B,Hl,Wl], int64
        reconstructed_flow = self.decode_from_indices(indices_decoded)
        return tf.cast(reconstructed_flow, tf.float32)

    @tf.function
    def encode_to_indices(self, flow_input):
       """ Encodes a flow map input to its spatial grid of VQ indices. """
       flow_input_f32 = tf.cast(flow_input, tf.float32)
       encoded = self.encoder(flow_input_f32, training=False)
       _, indices = self.vq_layer(encoded)
       return indices

    @tf.function
    def decode_from_indices(self, indices):
        """ Decodes a spatial grid of VQ indices back to a flow map. """
        flat_indices = tf.reshape(indices, [-1])
        quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.vq_layer.embeddings), flat_indices)
        latent_H, latent_W = self.encoder_output_shape[:2]
        batch_size = tf.shape(indices)[0]
        quantized_spatial_f32 = tf.reshape(quantized_flat, (batch_size, latent_H, latent_W, self.latent_dim))
        quantized_compute = tf.cast(quantized_spatial_f32, self.compute_dtype)
        reconstructed_flow = self.decoder(quantized_compute, training=False)
        return reconstructed_flow

    @tf.function
    def estimate_rate_bits(self, flow_input):
        """ Estimates the THEORETICAL bitrate using context model."""
        _, indices, encoded_y, quantized_ste = self(flow_input, training=False)
        rate_bits_per_element = self.hyperprior((encoded_y, quantized_ste, indices), training=False)
        num_latent_elements = tf.cast(tf.reduce_prod(tf.shape(indices)[1:]), tf.float32)
        total_estimated_bits = rate_bits_per_element * num_latent_elements
        return rate_bits_per_element, total_estimated_bits

# --- End AE Components ---


# --- Visualization Helpers & PSNR ---
# (make_color_wheel, flow_to_color, visualize_flow, calculate_psnr remain the same)
# ... (Assume these functions are defined correctly as in previous versions) ...

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Flow VQ Autoencoder with TFC HyperpriorSC or Generate Flow.')
    # ... (Define arguments as before) ...
    parser.add_argument('--mode', type=str, default='train_ae', choices=['generate_flow', 'train_ae'])
    parser.add_argument('--force_flow', action='store_true')
    parser.add_argument('--gen_batch_size', type=int, default=4)
    parser.add_argument('--ae_batch_size', type=int, default=AE_BATCH_SIZE)
    parser.add_argument('--ae_epochs', type=int, default=AE_EPOCHS)
    parser.add_argument('--lr', type=float, default=AE_LEARNING_RATE)
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA_RATE_DISTORTION)
    parser.add_argument('--hp_latent_dim', type=int, default=HYPERPRIOR_LATENT_DIM)
    parser.add_argument('--hp_filters', type=int, default=HYPERPRIOR_FILTERS)
    parser.add_argument('--ctx_filters', type=int, default=CONTEXT_FILTERS)
    parser.add_argument('--joint_filters', type=int, default=JOINT_FILTERS)
    parser.add_argument('--latent_dim', type=int, default=LATENT_DIM)
    parser.add_argument('--num_embeddings', type=int, default=NUM_EMBEDDINGS)
    parser.add_argument('--compress_demo', action='store_true', help="Run actual compression/decompression demo after training.")

    args = parser.parse_args()
    # ... (Override config with args) ...
    AE_BATCH_SIZE = args.ae_batch_size; AE_EPOCHS = args.ae_epochs; AE_LEARNING_RATE = args.lr
    BETA_RATE_DISTORTION = args.beta; LATENT_DIM = args.latent_dim; NUM_EMBEDDINGS = args.num_embeddings
    HYPERPRIOR_LATENT_DIM = args.hp_latent_dim; HYPERPRIOR_FILTERS = args.hp_filters
    CONTEXT_FILTERS = args.ctx_filters; JOINT_FILTERS = args.joint_filters

    print("\n--- Configuration ---"); # ... (print config) ... ; print("-" * 21 + "\n")

    if args.mode == 'generate_flow':
        # ... (Flow generation code) ...
        pass
    elif args.mode == 'train_ae':
        print(f"Selected Mode: Train VQ Autoencoder w/ TFC HyperpriorSC (Beta={BETA_RATE_DISTORTION})")
        print("Instantiating FlowVQAutoencoder...")
        flow_ae = FlowVQAutoencoder(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 2), latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS,
            commitment_cost=COMMITMENT_COST, beta_rate=BETA_RATE_DISTORTION, num_filters=HYPERPRIOR_FILTERS, # Reuse filters
            hyper_latent_dim=HYPERPRIOR_LATENT_DIM, hyper_filters=HYPERPRIOR_FILTERS,
            context_filters=CONTEXT_FILTERS, joint_filters=JOINT_FILTERS )
        print(f"Configuring Optimizer (LR={AE_LEARNING_RATE})...")
        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
             ae_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(ae_optimizer); print("Using LossScaleOptimizer.")
        reconstruction_loss_fn = tf.keras.losses.MeanSquaredError(name="mean_squared_error")
        print("Compiling Autoencoder model..."); flow_ae.compile(optimizer=ae_optimizer); print("Model compiled.")
        print("Building AE model with dummy input...")
        dummy_flow = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 2], dtype=tf.float32)
        try:
            _ = flow_ae(dummy_flow, training=False); print("FlowVQAutoencoder built successfully.")
            # Try summary without explicit hyperprior call first
            flow_ae.summary(expand_nested=True, line_length=120)
        except Exception as e: print(f"Error building/summarizing model: {e}", file=sys.stderr); import traceback; traceback.print_exc(); sys.exit(1)

        # Checkpoint Manager & Restore
        ae_ckpt = tf.train.Checkpoint(model=flow_ae, optimizer=ae_optimizer)
        ae_ckpt_manager = tf.train.CheckpointManager(ae_ckpt, AE_CHECKPOINT_DIR, max_to_keep=5)
        initial_epoch = 0
        # ... (Restore logic) ...

        # Dataset
        print(f"Preparing training dataset from: {FLOW_OUTPUT_DIR}")
        try:
            ae_train_dataset = configure_dataset_for_ae_from_flow(FLOW_OUTPUT_DIR, batch_size=AE_BATCH_SIZE)
            # ... (steps_per_epoch calculation) ...
            steps_per_epoch = None # Calculate steps if possible
        except Exception as e: print(f"Dataset Error: {e}", file=sys.stderr); sys.exit(1)

        # Callbacks
        # ... (Setup TensorBoard, ModelCheckpoint, EarlyStopping) ...
        ae_current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_sub_dir = (f"{ae_current_time}_beta{BETA_RATE_DISTORTION:.4f}" f"_lr{AE_LEARNING_RATE:.0e}_vq{NUM_EMBEDDINGS}_ld{LATENT_DIM}")
        ae_log_dir_epoch = os.path.join(AE_LOG_DIR, log_sub_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ae_log_dir_epoch, histogram_freq=1)
        print(f"Logging TensorBoard data to: {ae_log_dir_epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(AE_CHECKPOINT_DIR, 'ckpt-{epoch:04d}'), save_weights_only=False, save_freq='epoch', verbose=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='total_loss', patience=10, verbose=1, restore_best_weights=True)


        # Training
        print(f"\n--- Starting Training (Beta={BETA_RATE_DISTORTION}) ---")
        # ... (Dataset check) ...
        history = flow_ae.fit(
            ae_train_dataset, epochs=AE_EPOCHS, initial_epoch=initial_epoch,
            callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback],
            steps_per_epoch=steps_per_epoch)
        print("\nTraining finished.")

        # Save Final Model
        # ... (Save model logic) ...

        # --- Inference, Compression/Decompression Demo, Visualization ---
        print("\nRunning inference, compression demo, and visualization...")
        vis_output_dir = "./visualization_output_vqsc_tfc" # Specific dir
        os.makedirs(vis_output_dir, exist_ok=True)
        os.makedirs(COMPRESSED_OUTPUT_DIR, exist_ok=True)

        vis_img1_path = os.path.join(DATA_DIR, VISUALIZATION_SCENE, VISUALIZATION_SEQUENCE, 'im1.png')
        vis_flow_path = get_flow_output_path(vis_img1_path, DATA_DIR, FLOW_OUTPUT_DIR)

        if vis_flow_path is None or not os.path.exists(vis_flow_path):
             print(f"Error: Flow map for viz not found: {vis_flow_path}", file=sys.stderr)
        else:
            print(f"Loading original flow map for viz/compression from: {vis_flow_path}")
            try:
                original_flow_np = np.load(vis_flow_path).astype(np.float32)
                # ... (validation/resizing) ...
                original_flow_tf = tf.convert_to_tensor(original_flow_np[np.newaxis, ...], dtype=tf.float32)
                img1_np_vis = None # Load background image
                if os.path.exists(vis_img1_path): img1_np_vis = load_and_preprocess_image(vis_img1_path).numpy()
                else: img1_np_vis = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

                # Standard Reconstruction
                recon_tf, _, _, _ = flow_ae(original_flow_tf, training=False)
                recon_np = recon_tf[0].numpy()
                vis_file_base = f"vis_vqsc_tfc_beta{BETA_RATE_DISTORTION:.4f}_{VISUALIZATION_SCENE}_{VISUALIZATION_SEQUENCE}_im1"
                # ... (Visualize original and reconstructed) ...

                # Actual Compression Demo
                if args.compress_demo:
                    print("\nCompressing flow map using VQSC+TFC model...")
                    start_comp = time.time()
                    packed_strings, shapes = flow_ae.compress(original_flow_tf)
                    comp_time = time.time() - start_comp; print(f"Compression time: {comp_time:.4f}s")
                    z_strings_tf, indices_strings_tf = packed_strings
                    z_shape_tf, y_shape_tf = shapes
                    z_string_bytes = z_strings_tf.numpy()[0]
                    indices_string_bytes = indices_strings_tf.numpy()[0]
                    compressed_filename_base = f"vqsc_tfc_beta{BETA_RATE_DISTORTION:.4f}_{VISUALIZATION_SCENE}_{VISUALIZATION_SEQUENCE}_im1"
                    z_filepath = os.path.join(COMPRESSED_OUTPUT_DIR, f"{compressed_filename_base}_z.tfci")
                    idx_filepath = os.path.join(COMPRESSED_OUTPUT_DIR, f"{compressed_filename_base}_idx.tfci")
                    shapes_filepath = os.path.join(COMPRESSED_OUTPUT_DIR, f"{compressed_filename_base}_shapes.pkl")
                    try:
                        print(f"Saving compressed strings (z, indices) and shapes...")
                        with open(z_filepath, "wb") as f: f.write(z_string_bytes)
                        with open(idx_filepath, "wb") as f: f.write(indices_string_bytes)
                        shapes_to_save = {'z_shape': z_shape_tf.numpy(), 'y_shape': y_shape_tf.numpy()}
                        with open(shapes_filepath, 'wb') as f: pickle.dump(shapes_to_save, f)
                        z_bytes = os.path.getsize(z_filepath); idx_bytes = os.path.getsize(idx_filepath)
                        total_bytes = z_bytes + idx_bytes; bpp = total_bytes * 8 / (IMG_HEIGHT * IMG_WIDTH)
                        print(f" z bitstream size: {z_bytes} bytes"); print(f" indices bitstream size: {idx_bytes} bytes")
                        print(f" Total bitstream size: {total_bytes} bytes (~{total_bytes/1024:.2f} KB)"); print(f" Actual BPP (Context-Free Indices): {bpp:.4f}")
                    except Exception as e: print(f"Error saving compressed data: {e}", file=sys.stderr)

                    # Actual Decompression
                    print("\nDecompressing flow map using VQSC+TFC model...")
                    try:
                        with open(shapes_filepath, 'rb') as f: loaded_shapes = pickle.load(f)
                        loaded_z_shape = loaded_shapes['z_shape']; loaded_y_shape = loaded_shapes['y_shape']
                        with open(z_filepath, "rb") as f: loaded_z_string = f.read()
                        with open(idx_filepath, "rb") as f: loaded_idx_string = f.read()
                        loaded_z_strings_tf = tf.constant([loaded_z_string], dtype=tf.string); loaded_idx_strings_tf = tf.constant([loaded_idx_string], dtype=tf.string)
                        loaded_z_shape_tf = tf.constant(loaded_z_shape, dtype=tf.int32); loaded_y_shape_tf = tf.constant(loaded_y_shape, dtype=tf.int32)
                        start_decomp = time.time()
                        decompressed_flow_tf = flow_ae.decompress([loaded_z_strings_tf, loaded_idx_strings_tf], [loaded_z_shape_tf, loaded_y_shape_tf])
                        decomp_time = time.time() - start_decomp; decompressed_flow_np = decompressed_flow_tf[0].numpy(); print(f"Decompression time: {decomp_time:.4f}s")
                        vis_filename_prefix_decomp = os.path.join(vis_output_dir, f"{vis_file_base}_ACTUAL_DECOMP")
                        print("\nVisualizing ACTUAL DECOMPRESSED flow:"); visualize_flow(img1_np_vis, decompressed_flow_np, filename_prefix=vis_filename_prefix_decomp)
                        epe_decomp = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(original_flow_tf - decompressed_flow_tf), axis=-1) + 1e-9)).numpy()
                        mse_decomp = reconstruction_loss_fn(original_flow_tf, decompressed_flow_tf).numpy()
                        print(f"\n--- Performance of Actual Decompressed vs Original ---")
                        print(f"  - Decompressed EPE: {epe_decomp:.4f} pixels"); print(f"  - Decompressed MSE: {mse_decomp:.6f}")
                        print(f"  - Actual BPP (Context-Free Indices): {bpp:.4f}")
                    except Exception as e: print(f"Error during TFC decompression: {e}", file=sys.stderr); import traceback; traceback.print_exc()

                # Final Performance Estimation
                rate_bpp_element, total_bits_est = flow_ae.estimate_rate_bits(original_flow_tf)
                rate_bpp_element_np = rate_bpp_element.numpy(); total_bits_est_np = total_bits_est.numpy()
                epe = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(original_flow_tf - recon_tf), axis=-1) + 1e-9)).numpy()
                mse = reconstruction_loss_fn(original_flow_tf, recon_tf).numpy()
                print(f"\n--- Final Performance Estimate on Sample (Beta={BETA_RATE_DISTORTION}) ---")
                print(f"  - Recon EPE (Direct): {epe:.4f} pixels"); print(f"  - Recon MSE (Direct): {mse:.6f}")
                print(f"  - Estimated Rate (Contextual): {rate_bpp_element_np:.4f} bits/latent element")
                num_pixels = IMG_HEIGHT * IMG_WIDTH
                if num_pixels > 0: rate_bpp_input = total_bits_est_np / num_pixels; print(f"  - Estimated Rate (Contextual): {rate_bpp_input:.4f} bits/input pixel")
                print(f"  - Total Estimated Bits (Contextual): {total_bits_est_np:.0f} bits (~{total_bits_est_np/8/1024:.2f} KB)")
                if args.compress_demo and 'bpp' in locals(): print(f"  - Actual BPP (Context-Free Indices): {bpp:.4f}")

            except Exception as e: print(f"Error during visualization/inference: {e}", file=sys.stderr); import traceback; traceback.print_exc()
    else: print(f"Error: Unknown mode: {args.mode}", file=sys.stderr); parser.print_help(); sys.exit(1)

    print("\nScript Complete.")