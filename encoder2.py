# -*- coding: utf-8 -*-
import tensorflow as tf
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

# Prevent TensorFlow from allocating all GPU memory unnecessarily
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# --- Configuration ---
# RAFT specific
IMG_HEIGHT = 384
IMG_WIDTH = 512
NUM_ITERATIONS = 8
RAFT_CHECKPOINT_PATH = './raft_checkpoints/raft_final_weights.weights.h5'

# Autoencoder specific
AE_BATCH_SIZE = 2 # Reduced batch size from previous step
AE_EPOCHS = 200
AE_LEARNING_RATE = 1e-4
AE_CHECKPOINT_DIR = './flow_ae_checkpoints'
AE_LOG_DIR = './flow_ae_logs'
LATENT_DIM = 64
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25

# Shared / General
DATA_DIR = './video_data' # Base directory containing scene folders (imX.png)
# !!! New: Directory to store/load pre-computed flow maps !!!
FLOW_OUTPUT_DIR = './flow_data_generated'
# Visualization settings (using scene/sequence from DATA_DIR to find corresponding flow)
VISUALIZATION_SCENE = '00081' # Example scene name within DATA_DIR
VISUALIZATION_SEQUENCE = '0001'     # Example sequence ID within the scene
# Checkpoint saving frequency
SAVE_FREQ = None

# Ensure directories exist
os.makedirs(AE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AE_LOG_DIR, exist_ok=True)
os.makedirs(FLOW_OUTPUT_DIR, exist_ok=True) # Create flow output dir

# --- Mixed Precision ---
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16 precision.")
except Exception as e:
    print(f"Could not set mixed precision: {e}. Using default float32.")
    tf.keras.mixed_precision.set_global_policy('float32')

# --- 1. RAFT Model Implementation ---
# --- [Start of RAFT Model Definitions - UNCHANGED] ---
# Basic Residual Block
def BasicBlock(filters, stride=1):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
    ])

def DownsampleBlock(filters, stride=2):
     return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32),
        tf.keras.layers.ReLU(),
    ])

class FeatureEncoder(tf.keras.Model):
    def __init__(self, name='feature_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32)
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = DownsampleBlock(64, stride=1)
        self.layer2 = DownsampleBlock(96, stride=2)
        self.layer3 = DownsampleBlock(128, stride=2)
        self.conv_out = tf.keras.layers.Conv2D(256, kernel_size=1)
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x)
        return x

class ContextEncoder(tf.keras.Model):
    def __init__(self, name='context_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32)
        self.relu1 = tf.keras.layers.ReLU()
        self.layer1 = DownsampleBlock(64, stride=1)
        self.layer2 = DownsampleBlock(96, stride=2)
        self.layer3 = DownsampleBlock(128, stride=2)
        self.conv_out = tf.keras.layers.Conv2D(128, kernel_size=1)
    def call(self, x):
        x = self.conv1(x); x = self.norm1(x); x = self.relu1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.conv_out(x)
        return x

class ConvGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_filters, input_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters; self.input_filters = input_filters; self.kernel_size = kernel_size
        self.state_size = tf.TensorShape([None, None, hidden_filters])
        self.conv_update = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_reset = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_candidate = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='tanh', kernel_initializer='glorot_uniform')
    def build(self, input_shape): pass
    def call(self, inputs, states):
        h_prev = states[0]
        combined_input_h = tf.concat([inputs, h_prev], axis=-1)
        update_gate = self.conv_update(combined_input_h); reset_gate = self.conv_reset(combined_input_h)
        combined_input_reset_h = tf.concat([inputs, reset_gate * h_prev], axis=-1)
        candidate_h = self.conv_candidate(combined_input_reset_h)
        new_h = (1. - update_gate) * h_prev + update_gate * candidate_h
        return new_h, [new_h]

class UpdateBlock(tf.keras.Model):
    def __init__(self, iterations, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='update_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iterations = iterations; self.hidden_dim = hidden_dim
        corr_feature_dim = (2 * corr_radius + 1)**2 * corr_levels; motion_encoder_output_dim = 32
        inp_dim = max(0, context_dim - hidden_dim); gru_input_total_dim = motion_encoder_output_dim + inp_dim
        self.motion_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 1, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(motion_encoder_output_dim, 3, padding='same', activation='relu')
            ], name='motion_encoder')
        self.gru_cell = ConvGRUCell(hidden_filters=hidden_dim, input_filters=gru_input_total_dim)
        self.flow_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(2, 3, padding='same')
            ], name='flow_head')
    def call(self, net, inp, corr_features, flow_init=None):
        shape_tensor = tf.shape(net); b, h, w = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        flow = tf.zeros([b, h, w, 2], dtype=tf.float32) if flow_init is None else tf.cast(flow_init, tf.float32)
        hidden_state = net; flow_predictions = []
        for _ in range(self.iterations):
            flow = tf.stop_gradient(flow)
            motion_input = tf.concat([corr_features, flow], axis=-1)
            motion_features = self.motion_encoder(motion_input)
            gru_input = tf.concat([motion_features, inp], axis=-1)
            hidden_state, [hidden_state] = self.gru_cell(gru_input, [hidden_state])
            delta_flow = tf.cast(self.flow_head(hidden_state), tf.float32)
            flow = flow + delta_flow; flow_predictions.append(flow)
        return flow_predictions

def build_correlation_volume(fmap1, fmap2, radius=4):
    compute_dtype = fmap1.dtype; fmap2 = tf.cast(fmap2, compute_dtype)
    batch_size, h, w, c = tf.shape(fmap1)[0], tf.shape(fmap1)[1], tf.shape(fmap1)[2], tf.shape(fmap1)[3]
    pad_size = radius; fmap2_padded = tf.pad(fmap2, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')
    gy, gx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij'); coords_base = tf.stack([gy, gx], axis=-1)
    coords_base = tf.cast(coords_base, tf.int32); coords_base = tf.expand_dims(tf.expand_dims(coords_base, 0), -2)
    coords_base = tf.tile(coords_base, [batch_size, 1, 1, 1, 1])
    dy, dx = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1), indexing='ij')
    delta = tf.stack([dy, dx], axis=-1); num_neighbors = (2*radius+1)**2
    delta = tf.reshape(delta, [1, 1, 1, num_neighbors, 2]); delta = tf.cast(delta, tf.int32)
    lookup_coords = coords_base + delta + pad_size; batch_indices = tf.range(batch_size)
    batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1, 1]); batch_indices = tf.tile(batch_indices, [1, h, w, num_neighbors])
    lookup_indices = tf.stack([batch_indices, lookup_coords[..., 0], lookup_coords[..., 1]], axis=-1)
    fmap2_neighbors = tf.gather_nd(fmap2_padded, lookup_indices); fmap1_expanded = tf.expand_dims(fmap1, axis=3)
    correlation = tf.reduce_sum(fmap1_expanded * fmap2_neighbors, axis=-1)
    correlation_float32 = tf.cast(correlation, tf.float32)
    correlation_normalized = correlation_float32 / tf.maximum(tf.cast(c, tf.float32), 1e-6)
    return correlation_normalized

class RAFT(tf.keras.Model):
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_iterations = num_iterations; self.hidden_dim = hidden_dim; self.context_dim = context_dim
        self.corr_levels = corr_levels; self.corr_radius = corr_radius
        self.feature_encoder = FeatureEncoder(); self.context_encoder = ContextEncoder()
        self.update_block = UpdateBlock(iterations=num_iterations, hidden_dim=hidden_dim, context_dim=context_dim, corr_levels=corr_levels, corr_radius=corr_radius)
    @tf.function
    def upsample_flow(self, flow, target_height, target_width):
        flow = tf.cast(flow, tf.float32); shape_tensor = tf.shape(flow)
        b, h_low, w_low = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        h_low_safe = tf.maximum(tf.cast(h_low, tf.float32), 1.0); w_low_safe = tf.maximum(tf.cast(w_low, tf.float32), 1.0)
        scale_factor_h = tf.cast(target_height, tf.float32) / h_low_safe; scale_factor_w = tf.cast(target_width, tf.float32) / w_low_safe
        flow_upsampled = tf.image.resize(flow, [target_height, target_width], method='bilinear')
        u = flow_upsampled[..., 0] * scale_factor_w; v = flow_upsampled[..., 1] * scale_factor_h
        flow_scaled = tf.stack([u, v], axis=-1); return flow_scaled
    def call(self, inputs, training=False):
        image1, image2 = inputs; target_height = tf.shape(image1)[1]; target_width = tf.shape(image1)[2]
        fmap1 = self.feature_encoder(image1); fmap2 = self.feature_encoder(image2); context_fmap = self.context_encoder(image1)
        split_sizes = [self.hidden_dim, max(0, self.context_dim - self.hidden_dim)]
        if sum(split_sizes) != self.context_dim: raise ValueError(f"Context split sizes {split_sizes} do not sum")
        net, inp = tf.split(context_fmap, split_sizes, axis=-1)
        net = tf.tanh(net); inp = tf.nn.relu(inp)
        corr_features = build_correlation_volume(fmap1, fmap2, radius=self.corr_radius)
        flow_predictions_low_res = self.update_block(net, inp, corr_features, flow_init=None)
        flow_predictions_upsampled = [self.upsample_flow(flow_lr, target_height, target_width) for flow_lr in flow_predictions_low_res]
        return flow_predictions_upsampled
# --- [End of RAFT Model Definitions] ---


# --- 2. Dataset Loading and Flow Generation/Loading Functions ---

def load_and_preprocess_image(path):
    """Loads and preprocesses a single image."""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
        img = tf.image.convert_image_dtype(img, tf.float32)
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return img
    except Exception as e:
        tf.print(f"Error loading image {path}: {e}", output_stream=sys.stderr)
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

def find_image_pairs(data_dir):
    """Finds consecutive image pairs (imX.png, imX+1.png) in the dataset structure."""
    # --- Define helper function locally ---
    def parse_frame_num(filename):
        """Extracts frame number from filename like imX.png"""
        basename = os.path.basename(filename)
        match = re.search(r'im(\d+)\.png', basename)
        return int(match.group(1)) if match else -1
    # --- End local helper definition ---

    image_pairs = []
    print(f"Scanning dataset structure for image pairs in: {data_dir}")
    scene_folders = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not scene_folders:
        print(f"Warning: No scene folders found in {data_dir}.")
        return image_pairs

    for scene_path in tqdm(scene_folders, desc="Scanning Scenes"):
        sequence_folders = sorted([os.path.join(scene_path, s) for s in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, s)) and s.isdigit()])
        for seq_path in sequence_folders:
            frame_pattern = os.path.join(seq_path, 'im*.png')
            frame_files = glob(frame_pattern)
            try:
                valid_frames = [(f, parse_frame_num(f)) for f in frame_files if parse_frame_num(f) != -1]
                if len(valid_frames) < 2: continue
                sorted_frames_with_nums = sorted(valid_frames, key=lambda item: item[1])
                sorted_frames = [item[0] for item in sorted_frames_with_nums]
            except Exception as e:
                 print(f"\nWarning: Error processing frames in {seq_path}: {e}. Skipping.")
                 continue

            for i in range(len(sorted_frames) - 1):
                frame1_path = sorted_frames[i]
                frame2_path = sorted_frames[i+1]
                frame1_num = sorted_frames_with_nums[i][1]
                frame2_num = sorted_frames_with_nums[i+1][1]

                if frame2_num == frame1_num + 1:
                    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
                        image_pairs.append((frame1_path, frame2_path))

    print(f"Found {len(image_pairs)} consecutive image pairs.")
    return image_pairs

def get_flow_output_path(frame1_path, data_dir, flow_output_dir):
    """Determines the output path for the flow map corresponding to frame1_path."""
    rel_path = os.path.relpath(frame1_path, data_dir)
    base, _ = os.path.splitext(rel_path)
    flow_filename = f"flow_{os.path.basename(base)}.npy"
    output_path = os.path.join(flow_output_dir, os.path.dirname(rel_path), flow_filename)
    return output_path

def generate_flow_maps(data_dir, flow_output_dir, raft_model, batch_size=4, force_regenerate=False):
    """Generates flow maps using RAFT and saves them to flow_output_dir."""
    print("\n--- Starting Flow Generation Phase ---")
    image_pairs = find_image_pairs(data_dir)
    if not image_pairs:
        print("No image pairs found. Skipping flow generation.")
        return

    generated_count = 0; skipped_count = 0; error_count = 0
    img1_paths = [p[0] for p in image_pairs]; img2_paths = [p[1] for p in image_pairs]
    path_ds = tf.data.Dataset.from_tensor_slices((img1_paths, img2_paths))
    image_ds = path_ds.map(lambda p1, p2: (load_and_preprocess_image(p1), load_and_preprocess_image(p2)),
                           num_parallel_calls=tf.data.AUTOTUNE)
    batched_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    output_flow_paths = [get_flow_output_path(p[0], data_dir, flow_output_dir) for p in image_pairs]

    pair_idx = 0
    for img1_batch_tf, img2_batch_tf in tqdm(batched_ds, desc="Generating Flow Maps", total=len(image_pairs)//batch_size + (1 if len(image_pairs)%batch_size else 0)):
        try:
            flow_predictions_list = raft_model([img1_batch_tf, img2_batch_tf], training=False)
            final_flow_batch_tf = flow_predictions_list[-1]
            final_flow_batch_np = final_flow_batch_tf.numpy()

            current_batch_size = tf.shape(img1_batch_tf)[0].numpy()
            for i in range(current_batch_size):
                output_path = output_flow_paths[pair_idx]
                if not force_regenerate and os.path.exists(output_path):
                    skipped_count += 1; pair_idx += 1; continue

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, final_flow_batch_np[i].astype(np.float32))
                generated_count += 1; pair_idx += 1
        except Exception as e:
            print(f"\nError during flow generation for batch starting at index {pair_idx}: {e}")
            current_batch_size_on_error = tf.shape(img1_batch_tf)[0].numpy() if 'img1_batch_tf' in locals() else batch_size
            error_count += current_batch_size_on_error; pair_idx += current_batch_size_on_error

    print("\n--- Flow Generation Summary ---")
    print(f"Flow maps generated: {generated_count}"); print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors encountered (estimated pairs): {error_count}"); print(f"Flow maps saved in: {flow_output_dir}")
    print("-" * 30)

    
def load_flow_map(flow_path_tensor):
    """Loads a .npy flow map using tf.py_function."""
    def _load_npy(path_tensor): # Renamed input for clarity
        # --- FIX: Extract numpy bytes value from tensor ---
        path_bytes = path_tensor.numpy()
        # Decode the numpy bytes to a python string
        path_str = path_bytes.decode('utf-8')
        # --- End Fix ---
        try:
            flow_map = np.load(path_str)
            if flow_map.shape != (IMG_HEIGHT, IMG_WIDTH, 2):
                 print(f"Warning: Flow map {path_str} shape {flow_map.shape} != expected {(IMG_HEIGHT, IMG_WIDTH, 2)}. Resizing.", file=sys.stderr)
                 flow_map = cv2.resize(flow_map, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            return flow_map.astype(np.float32)
        except Exception as e:
            # It's good practice to print the path that failed
            print(f"Error loading flow map {path_str}: {e}", file=sys.stderr)
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.float32)

    flow = tf.py_function(
        _load_npy,
        [flow_path_tensor], # Input is still the tf.string tensor
        tf.float32
    )
    flow.set_shape([IMG_HEIGHT, IMG_WIDTH, 2])
    return flow

def configure_dataset_for_ae_from_flow(flow_dir, batch_size):
    """ Configures dataset for AE training by loading pre-computed flow maps (.npy) """
    print(f"Scanning for pre-computed flow maps (.npy) in: {flow_dir}")
    flow_map_paths = glob(os.path.join(flow_dir, '**', 'flow_im*.npy'), recursive=True)
    if not flow_map_paths:
        raise ValueError(f"No pre-computed flow maps (flow_im*.npy) found in {flow_dir} or subdirs. Run with '--mode generate_flow'.")
    print(f"Found {len(flow_map_paths)} potential flow maps.")

    flow_path_ds = tf.data.Dataset.from_tensor_slices(flow_map_paths)
    flow_ds = flow_path_ds.map(load_flow_map, num_parallel_calls=tf.data.AUTOTUNE)
    flow_ds = flow_ds.shuffle(buffer_size=min(len(flow_map_paths), 1000))
    flow_ds = flow_ds.batch(batch_size)
    flow_ds = flow_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Dataset configured to load pre-computed flow maps.")
    return flow_ds


# --- 3. Flow Autoencoder Model Implementation ---
# --- [Start of AE Model Definitions - UNCHANGED from previous working version] ---
# Vector Quantizer Layer
class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim; self.num_embeddings = num_embeddings; self.commitment_cost = commitment_cost
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable( initial_value=w_init(shape=(embedding_dim, num_embeddings), dtype="float32"), trainable=True, name="embeddings_vq", )
    def call(self, inputs):
        input_dtype = inputs.dtype; inputs_float32 = tf.cast(inputs, dtype=tf.float32)
        flat_inputs = tf.reshape(inputs_float32, [-1, self.embedding_dim])
        distances = ( tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True) - 2 * tf.matmul(flat_inputs, self.embeddings) + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True) )
        encoding_indices = tf.argmin(distances, axis=1)
        quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.embeddings), encoding_indices)
        input_shape = tf.shape(inputs); quantized_spatial = tf.reshape(quantized_flat, input_shape)
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized_spatial) - inputs_float32))
        q_latent_loss = tf.reduce_mean(tf.square(quantized_spatial - tf.stop_gradient(inputs_float32)))
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        self.add_loss(tf.cast(loss, dtype=tf.float32))
        difference_f32 = quantized_spatial - inputs_float32
        stopped_difference_f32 = tf.stop_gradient(difference_f32)
        stopped_difference_casted = tf.cast(stopped_difference_f32, input_dtype)
        quantized_ste = inputs + stopped_difference_casted
        return quantized_ste, encoding_indices

# Autoencoder Architecture
def build_encoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    x = tf.cast(inputs, tf.keras.mixed_precision.global_policy().compute_dtype)
    activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)
    x = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    latent = tf.keras.layers.Conv2D(latent_dim, kernel_size=1, padding='same', name='encoder_output')(x)
    return tf.keras.Model(inputs, latent, name='flow_encoder')

def build_decoder(latent_shape, latent_dim):
    latent_inputs = tf.keras.Input(shape=latent_shape, dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    x = latent_inputs; activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x); x = activation_fn(x)
    outputs = tf.keras.layers.Conv2DTranspose(2, kernel_size=3, padding='same', activation='linear', dtype='float32', name='decoder_output')(x)
    return tf.keras.Model(latent_inputs, outputs, name='flow_decoder')

class FlowVQAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, num_embeddings, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.input_flow_shape = input_shape; self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings; self.commitment_cost = commitment_cost
        self.encoder = build_encoder(input_shape, latent_dim)
        dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
        latent_output = self.encoder(dummy_input)
        self.latent_spatial_shape = tf.shape(latent_output)[1:3].numpy()
        print(f"Autoencoder Determined Latent Spatial Shape: {self.latent_spatial_shape}")
        self.latent_shape = (self.latent_spatial_shape[0], self.latent_spatial_shape[1], latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, name="vector_quantizer")
        self.decoder = build_decoder(self.latent_shape, latent_dim)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")
    @property
    def metrics(self): return [self.total_loss_tracker, self.recon_loss_tracker, self.vq_loss_tracker]
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs, training=training)
        quantized_ste, indices = self.vq_layer(encoded)
        reconstructed = self.decoder(quantized_ste, training=training)
        return reconstructed, indices
    def train_step(self, data):
        flow_target = data # Dataset now yields flow maps directly
        with tf.GradientTape() as tape:
            reconstructed_flow, _ = self(flow_target, training=True)
            recon_loss = reconstruction_loss_fn(flow_target, reconstructed_flow)
            vq_loss = sum(self.vq_layer.losses) # Sum VQ losses added in the layer
            total_loss = recon_loss + vq_loss
            scaled_loss = total_loss
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                 scaled_loss = self.optimizer.get_scaled_loss(total_loss)
        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
             gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in gradients]
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(valid_grads_and_vars)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        flow_target = data; reconstructed_flow, _ = self(flow_target, training=False)
        recon_loss = reconstruction_loss_fn(flow_target, reconstructed_flow)
        vq_loss = sum(self.vq_layer.losses)
        total_loss = recon_loss + vq_loss
        self.total_loss_tracker.update_state(total_loss); self.recon_loss_tracker.update_state(recon_loss); self.vq_loss_tracker.update_state(vq_loss)
        return {m.name: m.result() for m in self.metrics}
    def encode_to_indices(self, flow_input):
        flow_input_f32 = tf.cast(flow_input, tf.float32); encoded = self.encoder(flow_input_f32, training=False)
        encoded_f32 = tf.cast(encoded, tf.float32); flat_encoded = tf.reshape(encoded_f32, [-1, self.latent_dim])
        distances = ( tf.reduce_sum(flat_encoded**2, axis=1, keepdims=True) - 2 * tf.matmul(flat_encoded, self.vq_layer.embeddings) + tf.reduce_sum(self.vq_layer.embeddings**2, axis=0, keepdims=True) )
        indices = tf.argmin(distances, axis=1)
        batch_size = tf.shape(flow_input)[0]; indices_reshaped = tf.reshape(indices, (batch_size, self.latent_spatial_shape[0], self.latent_spatial_shape[1]))
        return indices_reshaped
    def decode_from_indices(self, indices):
        flat_indices = tf.reshape(indices, [-1]); quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.vq_layer.embeddings), flat_indices)
        quantized_spatial = tf.reshape(quantized_flat, (tf.shape(indices)[0], self.latent_spatial_shape[0], self.latent_spatial_shape[1], self.latent_dim))
        quantized_compute = tf.cast(quantized_spatial, self.compute_dtype)
        reconstructed_flow = self.decoder(quantized_compute, training=False)
        return reconstructed_flow
# --- [End of AE Model Definitions] ---


# --- 4. Visualization Helpers ---
# --- [Start of Visualization Helpers - UNCHANGED] ---
def make_color_wheel():
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6; ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3)); col = 0
    colorwheel[0:RY, 0] = 255; colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY); col += RY
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG); colorwheel[col:col+YG, 1] = 255; col += YG
    colorwheel[col:col+GC, 1] = 255; colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC); col += GC
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB)/CB); colorwheel[col:col+CB, 2] = 255; col += CB
    colorwheel[col:col+BM, 2] = 255; colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM); col += BM
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR)/MR); colorwheel[col:col+MR, 0] = 255
    return colorwheel.astype(np.uint8)

def flow_to_color(flow, convert_to_bgr=False):
    if flow is None: return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    UNKNOWN_FLOW_THRESH = 1e7; SMALL_FLOW = 1e-9
    if not isinstance(flow, np.ndarray) or flow.ndim != 3 or flow.shape[2] != 2:
        print(f"Warning: Invalid flow shape in flow_to_color: {flow.shape if isinstance(flow, np.ndarray) else type(flow)}. Black.")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    height, width, _ = flow.shape
    if height == 0 or width == 0: return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    img = np.zeros((height, width, 3), dtype=np.uint8); colorwheel = make_color_wheel(); ncols = colorwheel.shape[0]
    u, v = flow[..., 0], flow[..., 1]; u = np.nan_to_num(u); v = np.nan_to_num(v)
    mag = np.sqrt(u**2 + v**2); ang = np.arctan2(-v, -u) / np.pi; ang = (ang + 1.0) / 2.0
    valid_indices = np.isfinite(mag) & (np.abs(mag) < UNKNOWN_FLOW_THRESH)
    mag_max = np.max(mag[valid_indices]) if np.any(valid_indices) else 0.0
    if mag_max <= SMALL_FLOW: mag_max = SMALL_FLOW
    mag_norm = np.clip(mag / mag_max, 0, 1)
    fk = (ang * (ncols - 1)); k0 = np.floor(fk).astype(np.int32); k1 = (k0 + 1) % ncols; f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]; k0 = np.clip(k0, 0, ncols - 1); k1 = np.clip(k1, 0, ncols - 1)
        col0 = tmp[k0] / 255.0; col1 = tmp[k1] / 255.0; col = (1 - f) * col0 + f * col1
        col = 1 - mag_norm * (1 - col); img[:, :, i] = np.clip(np.floor(255.0 * col), 0, 255)
    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH) | ~valid_indices
    img[idx_unknown] = 0
    if convert_to_bgr: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def visualize_flow(image1_np, flow_pred_np, filename_prefix="flow_vis"):
    if image1_np is None or flow_pred_np is None: print("Warning: Cannot visualize flow, input image or flow is None."); return
    image1_np = np.asarray(image1_np); flow_pred_np = np.asarray(flow_pred_np)
    if image1_np.dtype == np.float32 or image1_np.dtype == np.float64: image1_np = np.clip(image1_np, 0, 1)
    image1_np_display = (image1_np * 255).astype(np.uint8) if image1_np.dtype != np.uint8 else image1_np
    if flow_pred_np.ndim != 3 or flow_pred_np.shape[-1] != 2: print(f"Warning: Invalid flow shape for visualization: {flow_pred_np.shape}. Skipping."); return
    h, w, _ = flow_pred_np.shape; img_h, img_w, _ = image1_np_display.shape
    if img_h != h or img_w != w: image1_np_display = cv2.resize(image1_np_display, (w, h), interpolation=cv2.INTER_LINEAR)
    output_dir = os.path.dirname(filename_prefix);
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    try: # Magnitude
        plt.figure(figsize=(10, 8)); u_finite, v_finite = np.nan_to_num(flow_pred_np[..., 0]), np.nan_to_num(flow_pred_np[..., 1])
        magnitude = np.sqrt(u_finite**2 + v_finite**2); mag_finite = magnitude[np.isfinite(magnitude)]
        vmin = np.percentile(mag_finite, 1) if mag_finite.size > 0 else 0; vmax = np.percentile(mag_finite, 99) if mag_finite.size > 0 else 1
        im = plt.imshow(magnitude, cmap='viridis', vmin=vmin, vmax=vmax); plt.colorbar(im, label='Flow Magnitude (pixels)')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Magnitude'); plt.axis('off'); plt.tight_layout()
        save_path = f"{filename_prefix}_magnitude.png"; plt.savefig(save_path, bbox_inches='tight'); plt.close(); print(f"Saved mag: {save_path}")
    except Exception as e: print(f"Error mag plot: {e}"); plt.close()
    try: # Vectors
        step = max(1, min(h, w) // 32); y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int); fx, fy = np.nan_to_num(flow_pred_np[y, x].T)
        plt.figure(figsize=(12, 9)); plt.imshow(image1_np_display)
        plt.quiver(x, y, fx, fy, color='red', scale=None, scale_units='xy', angles='xy', headwidth=5, headlength=6, width=0.0015, pivot='tail')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Vectors (Overlay)'); plt.axis('off'); plt.tight_layout()
        save_path = f"{filename_prefix}_vectors.png"; plt.savefig(save_path, bbox_inches='tight'); plt.close(); print(f"Saved vec: {save_path}")
    except Exception as e: print(f"Error vec plot: {e}"); plt.close()
    try: # Color
        flow_color_img = flow_to_color(flow_pred_np); plt.figure(figsize=(10, 8)); plt.imshow(flow_color_img)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow (Color)'); plt.axis('off'); plt.tight_layout()
        save_path = f"{filename_prefix}_color.png"; plt.savefig(save_path, bbox_inches='tight'); plt.close(); print(f"Saved color: {save_path}")
    except Exception as e: print(f"Error color plot: {e}"); plt.close()
# --- [End of Visualization Helpers] ---


# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Flow Autoencoder or Generate Flow Maps.')
    parser.add_argument('--mode', type=str, default='train_ae', choices=['generate_flow', 'train_ae'],
                        help='Operation mode: "generate_flow" to pre-compute flow maps, "train_ae" to train the autoencoder.')
    parser.add_argument('--force_flow', action='store_true',
                        help='Force regeneration of flow maps even if they exist (only applies to generate_flow mode).')
    parser.add_argument('--gen_batch_size', type=int, default=4,
                        help='Batch size for flow generation (adjust based on GPU memory).')

    args = parser.parse_args()

    # --- Phase 1: Flow Generation ---
    if args.mode == 'generate_flow':
        print("Selected Mode: Generate Flow Maps")
        # Load RAFT model for generation
        print("Loading RAFT model...")
        # Assign to a distinct variable name if needed elsewhere
        raft_generator_model = RAFT(num_iterations=NUM_ITERATIONS)
        dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        try:
            _ = raft_generator_model([dummy_img1, dummy_img2], training=False) # Build the specific instance
            print("RAFT model built for generation.")
            if os.path.exists(RAFT_CHECKPOINT_PATH):
                print(f"Attempting to load RAFT weights from {RAFT_CHECKPOINT_PATH}...")
                raft_generator_model.load_weights(RAFT_CHECKPOINT_PATH) # Load into this instance
                print(f"Successfully loaded RAFT weights.")
            else:
                 print(f"!!! ERROR: RAFT checkpoint file not found at {RAFT_CHECKPOINT_PATH}. Cannot generate flow. !!!")
                 sys.exit(1) # Exit if weights are missing
        except Exception as e:
            print(f"Error building or loading RAFT model for generation: {e}")
            sys.exit(1)

        raft_generator_model.trainable = False # Freeze this instance
        print("RAFT model frozen for generation.")

        # Run generation function
        generate_flow_maps(DATA_DIR, FLOW_OUTPUT_DIR, raft_generator_model,
                           batch_size=args.gen_batch_size, force_regenerate=args.force_flow)

    # --- Phase 2: Autoencoder Training ---
    elif args.mode == 'train_ae':
        print("Selected Mode: Train Autoencoder")

        # No need to load RAFT model here for training AE on pre-computed flow

        # Instantiate the Autoencoder
        print("Instantiating Flow VQ Autoencoder...")
        flow_ae = FlowVQAutoencoder(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 2),
            latent_dim=LATENT_DIM,
            num_embeddings=NUM_EMBEDDINGS,
            commitment_cost=COMMITMENT_COST
        )

        # Optimizer
        print("Configuring Optimizer...")
        ae_optimizer = tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE)

        # Loss Function (defined for clarity, used in train_step)
        reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

        # Compile the Autoencoder model
        print("Compiling Autoencoder model...")
        flow_ae.compile(optimizer=ae_optimizer) # Loss is implicitly handled by train_step metrics

        # Build the model (important AFTER compile when using compile/fit)
        dummy_flow = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 2], dtype=tf.float32)
        try:
            _ = flow_ae(dummy_flow) # Call the model to build layers
            print("Flow Autoencoder built and compiled.")
            flow_ae.summary()
        except Exception as e:
             print(f"Error building compiled AE model: {e}")
             sys.exit(1)

        # Checkpoint Manager
        ae_ckpt = tf.train.Checkpoint(model=flow_ae, optimizer=ae_optimizer)
        ae_ckpt_manager = tf.train.CheckpointManager(ae_ckpt, AE_CHECKPOINT_DIR, max_to_keep=5)

        # Restore AE Checkpoint
        initial_epoch = 0
        if ae_ckpt_manager.latest_checkpoint:
            print(f"Restoring AE checkpoint from {ae_ckpt_manager.latest_checkpoint}...")
            status = ae_ckpt.restore(ae_ckpt_manager.latest_checkpoint)
            try:
                status.assert_existing_objects_matched().expect_partial()
                print("AE Checkpoint restored successfully.")
                try:
                    # Try to get epoch number from filename ckpt-N
                    initial_epoch = int(re.findall(r'ckpt-(\d+)', ae_ckpt_manager.latest_checkpoint)[0]) -1 # epoch is 0-based index
                    initial_epoch = max(0, initial_epoch) # Ensure not negative
                    print(f"Resuming AE training from Epoch {initial_epoch+1}") # Display 1-based epoch
                except (IndexError, ValueError, TypeError):
                    print("Could not determine resume epoch from checkpoint filename. Starting from epoch 0.")
                    initial_epoch = 0
            except AssertionError as e:
                print(f"Warning: AE Checkpoint restoration issue: {e}. Training from scratch.")
                initial_epoch = 0
        else:
            print("No AE checkpoint found, initializing AE from scratch.")
            initial_epoch = 0

        # --- Prepare Dataset from Pre-computed Flow ---
        print(f"Preparing Autoencoder training dataset from pre-computed flow...")
        try:
            ae_train_dataset = configure_dataset_for_ae_from_flow(FLOW_OUTPUT_DIR, batch_size=AE_BATCH_SIZE)
        except ValueError as e:
            print(f"\nError creating dataset: {e}")
            print("Please ensure flow maps have been generated using '--mode generate_flow'.")
            sys.exit(1)
        except Exception as e:
             print(f"\nUnexpected error creating dataset: {e}")
             sys.exit(1)

        # Calculate steps per epoch
        steps_per_epoch = None
        try:
            ae_total_batches_per_epoch = tf.data.experimental.cardinality(ae_train_dataset)
            if ae_total_batches_per_epoch != tf.data.experimental.UNKNOWN_CARDINALITY:
                 steps_per_epoch = ae_total_batches_per_epoch.numpy()
                 if steps_per_epoch == 0:
                      print("\nError: Dataset created successfully but contains 0 elements.")
                      print(f"Check if .npy files exist in {FLOW_OUTPUT_DIR} and subdirectories.")
                      sys.exit(1)
                 print(f"Dataset size determined: {steps_per_epoch} steps per epoch.")
            else:
                 print("Could not determine dataset size. Steps per epoch unknown.")
        except Exception as e:
             print(f"Error determining dataset cardinality: {e}")

        # Callbacks
        ae_current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ae_log_dir_epoch = os.path.join(AE_LOG_DIR, ae_current_time + f"_start_epoch_{initial_epoch+1}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ae_log_dir_epoch, histogram_freq=1)
        print(f"Logging AE TensorBoard data to: {ae_log_dir_epoch}")

        class CheckpointManagerCallback(tf.keras.callbacks.Callback):
            def __init__(self, checkpoint_manager):
                super().__init__()
                self.checkpoint_manager = checkpoint_manager
            def on_epoch_end(self, epoch, logs=None):
                # Checkpoint number should align with epoch number (Keras uses 0-based)
                # Save as ckpt-(epoch+1) to be consistent with common practice
                save_path = self.checkpoint_manager.save(checkpoint_number=epoch + 1)
                print(f"\nEpoch {epoch+1}: Saved TF Checkpoint to {save_path}")
        checkpoint_manager_callback = CheckpointManagerCallback(ae_ckpt_manager)

        # --- Start Training ---
        print(f"\n--- Starting Autoencoder Training ---")
        print(f"Training from epoch {initial_epoch} up to {AE_EPOCHS}...") # User sees 1-based
        history = flow_ae.fit(
            ae_train_dataset,
            epochs=AE_EPOCHS,
            initial_epoch=initial_epoch, # Keras fit uses 0-based initial epoch
            callbacks=[tensorboard_callback, checkpoint_manager_callback],
            steps_per_epoch=steps_per_epoch
        )
        print("\nAutoencoder Training finished.")

        # --- Save Final Weights ---
        print("Saving final Autoencoder weights...")
        final_ae_weights_path = os.path.join(AE_CHECKPOINT_DIR, 'flow_ae_final_weights.weights.h5')
        try:
            flow_ae.save_weights(final_ae_weights_path)
            print(f"Final AE weights saved to {final_ae_weights_path}")
        except Exception as e:
            print(f"Error saving final AE weights: {e}")

        # --- Inference and Visualization ---
        print("\nRunning Autoencoder inference and visualization...")
        vis_output_dir = "./visualization_output"
        os.makedirs(vis_output_dir, exist_ok=True)

        # Find original image path for visualization context
        # Assuming visualization uses the first frame 'im1.png'
        vis_img1_path = os.path.join(DATA_DIR, VISUALIZATION_SCENE, VISUALIZATION_SEQUENCE, 'im1.png')
        # Construct the corresponding pre-computed flow path (flow generated from im1.png and im2.png)
        vis_flow_path = get_flow_output_path(vis_img1_path, DATA_DIR, FLOW_OUTPUT_DIR) # Path based on im1.png

        if not os.path.exists(vis_flow_path):
             print(f"Error: Pre-computed flow map for visualization not found at {vis_flow_path}")
             print(f"Please check VISUALIZATION_SCENE ('{VISUALIZATION_SCENE}'), VISUALIZATION_SEQUENCE ('{VISUALIZATION_SEQUENCE}'), FLOW_OUTPUT_DIR ('{FLOW_OUTPUT_DIR}') or run flow generation.")
        else:
            print(f"Loading original flow map for visualization from: {vis_flow_path}")
            try:
                original_flow_np = np.load(vis_flow_path)
                if original_flow_np.shape != (IMG_HEIGHT, IMG_WIDTH, 2):
                     print(f"Warning: Loaded viz flow map shape {original_flow_np.shape} != expected. Resizing.")
                     original_flow_np = cv2.resize(original_flow_np, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                original_flow_tf = tf.convert_to_tensor(original_flow_np[np.newaxis, ...], dtype=tf.float32)

                if os.path.exists(vis_img1_path):
                     img1_inf = load_and_preprocess_image(vis_img1_path)
                     img1_np_vis = img1_inf.numpy()
                else:
                     print(f"Warning: Image file {vis_img1_path} not found for visualization background.")
                     img1_np_vis = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

                print("Compressing and reconstructing flow with Autoencoder...")
                start_ae_inf = time.time()
                reconstructed_flow_tf, _ = flow_ae(original_flow_tf, training=False)
                ae_inf_time = time.time() - start_ae_inf
                reconstructed_flow_np = reconstructed_flow_tf[0].numpy()
                print(f"Autoencoder inference time: {ae_inf_time:.3f}s")

                vis_file_base = f"vis_ae_{VISUALIZATION_SCENE}_{VISUALIZATION_SEQUENCE}_im1"
                vis_filename_prefix_orig = os.path.join(vis_output_dir, f"{vis_file_base}_ORIGINAL_PRECOMP")
                print("\nVisualizing ORIGINAL flow (Loaded from .npy):")
                visualize_flow(img1_np_vis, original_flow_np, filename_prefix=vis_filename_prefix_orig)

                vis_filename_prefix_recon = os.path.join(vis_output_dir, f"{vis_file_base}_RECONSTRUCTED")
                print("\nVisualizing RECONSTRUCTED flow (AE output):")
                visualize_flow(img1_np_vis, reconstructed_flow_np, filename_prefix=vis_filename_prefix_recon)

                epe_map = tf.sqrt(tf.reduce_sum(tf.square(original_flow_tf - reconstructed_flow_tf), axis=-1) + 1e-8)
                epe = tf.reduce_mean(epe_map).numpy()
                mse = reconstruction_loss_fn(original_flow_tf, reconstructed_flow_tf).numpy()
                print(f"\nReconstruction Quality on this sample:")
                print(f"  - EPE: {epe:.4f} pixels"); print(f"  - MSE: {mse:.6f}")

                print("\nSimulating bitstream generation:")
                indices = flow_ae.encode_to_indices(original_flow_tf)
                indices_np = indices[0].numpy()
                unique_indices, counts = np.unique(indices_np, return_counts=True)
                print(f"  - Encoded flow to latent indices of shape: {indices_np.shape}")
                print(f"  - Unique indices used: {len(unique_indices)} / {flow_ae.num_embeddings}")
                probabilities = counts / indices_np.size
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
                theoretical_min_bits = entropy * indices_np.size
                bits_per_index = np.ceil(np.log2(flow_ae.num_embeddings))
                raw_bits = indices_np.size * bits_per_index
                print(f"  - Bits per index (fixed): {bits_per_index:.2f} bits"); print(f"  - Raw bit cost (fixed): {raw_bits:.0f} bits (~{raw_bits/8/1024:.2f} KB)")
                print(f"  - Avg bits per index (entropy): {entropy:.3f} bits"); print(f"  - Min bit cost (entropy): {theoretical_min_bits:.0f} bits (~{theoretical_min_bits/8/1024:.2f} KB)")
                print(f"  - (Actual size depends on entropy coder efficiency)")

            except Exception as e:
                print(f"Error during visualization/inference: {e}")

    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()

    print("\nScript Complete.")