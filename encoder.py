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
# RAFT specific (used for data generation)
IMG_HEIGHT = 384
IMG_WIDTH = 512
NUM_ITERATIONS = 8 # RAFT iterations for flow generation
# !!! IMPORTANT: Set the path to your trained RAFT weights !!!
RAFT_CHECKPOINT_PATH = './raft_checkpoints/raft_final_weights.weights.h5' # Adjust if needed

# Autoencoder specific
AE_BATCH_SIZE = 2 # Batch size for autoencoder training
AE_EPOCHS = 200 # Epochs for autoencoder training
AE_LEARNING_RATE = 1e-4 # Learning rate for autoencoder
AE_CHECKPOINT_DIR = './flow_ae_checkpoints' # Directory for AE checkpoints
AE_LOG_DIR = './flow_ae_logs' # Directory for AE TensorBoard logs
LATENT_DIM = 64      # Number of channels in the bottleneck (embedding dimension)
NUM_EMBEDDINGS = 512 # Size of the VQ codebook (number of discrete codes)
COMMITMENT_COST = 0.25 # Weight for VQ commitment loss

# Shared / General
# !!! IMPORTANT: Update DATA_DIR to your video data folder !!!
DATA_DIR = './video_data' # Base directory containing scene folders
VISUALIZATION_SCENE = 'scene_example_1' # Example scene name within DATA_DIR for visualization
VISUALIZATION_SEQUENCE = '0001'     # Example sequence ID within the scene for visualization
# --- Frequency for saving checkpoints ---
SAVE_FREQ = None # Set to an integer (e.g., 1000) to save every N steps, None to save every epoch

# Ensure directories exist
os.makedirs(AE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AE_LOG_DIR, exist_ok=True)

# --- Mixed Precision Setup (Optional but recommended) ---
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Using mixed_float16 precision.")
except Exception as e:
    print(f"Could not set mixed precision: {e}. Using default float32.")
    tf.keras.mixed_precision.set_global_policy('float32')


# --- 1. RAFT Model Implementation (Copied from original script) ---
# Need the RAFT model definition to load weights and generate flow
# (Keep all RAFT classes: BasicBlock, DownsampleBlock, FeatureEncoder, ContextEncoder, ConvGRUCell, UpdateBlock, RAFT, build_correlation_volume, upsample_flow)
# --- [Start of RAFT Model Definitions] ---
# Basic Residual Block
def BasicBlock(filters, stride=1):
    # Layer normalization/instance normalization can be float32 even in mixed precision
    # Usually okay, but check TF docs if issues arise.
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False),
        tfa.layers.InstanceNormalization(dtype=tf.float32), # Specify dtype for norm layers
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
        tf.keras.layers.ReLU(), # Add ReLU after second conv too
    ])


# Feature Encoder (f-net)
class FeatureEncoder(tf.keras.Model):
    def __init__(self, name='feature_encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)
        self.norm1 = tfa.layers.InstanceNormalization(dtype=tf.float32)
        self.relu1 = tf.keras.layers.ReLU()

        self.layer1 = DownsampleBlock(64, stride=1) # Output: H/2, W/2
        self.layer2 = DownsampleBlock(96, stride=2) # Output: H/4, W/4
        self.layer3 = DownsampleBlock(128, stride=2) # Output: H/8, W/8

        self.conv_out = tf.keras.layers.Conv2D(256, kernel_size=1) # Final projection

    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Shape: (B, H/8, W/8, 128)

        x = self.conv_out(x) # Shape: (B, H/8, W/8, 256)
        return x

# Context Encoder (c-net)
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
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # Shape: (B, H/8, W/8, 128)

        x = self.conv_out(x) # Shape: (B, H/8, W/8, 128)
        return x

# Convolutional GRU Cell
class ConvGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_filters, input_filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_filters = hidden_filters
        self.input_filters = input_filters
        self.kernel_size = kernel_size
        # TF will infer spatial dimensions, state needs hidden_filters channels
        self.state_size = tf.TensorShape([None, None, hidden_filters])

        # Gates and candidate calculation. Use sigmoid/tanh which handle mixed precision well.
        self.conv_update = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_reset = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')
        self.conv_candidate = tf.keras.layers.Conv2D(hidden_filters, kernel_size, padding='same', activation='tanh', kernel_initializer='glorot_uniform')

    def build(self, input_shape):
        # input_shape is [(batch, H, W, input_filters), (batch, H, W, hidden_filters)] (inputs, states)
        # Conv2D layers create weights in __init__ or on first call.
        pass

    def call(self, inputs, states):
        h_prev = states[0] # GRU has one state tensor
        combined_input_h = tf.concat([inputs, h_prev], axis=-1)
        update_gate = self.conv_update(combined_input_h)
        reset_gate = self.conv_reset(combined_input_h)
        combined_input_reset_h = tf.concat([inputs, reset_gate * h_prev], axis=-1)
        candidate_h = self.conv_candidate(combined_input_reset_h)
        new_h = (1. - update_gate) * h_prev + update_gate * candidate_h
        return new_h, [new_h] # Output and new state (must be a list)

# Motion Encoder and Update Block
class UpdateBlock(tf.keras.Model):
    def __init__(self, iterations, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='update_block', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iterations = iterations
        self.hidden_dim = hidden_dim
        corr_feature_dim = (2 * corr_radius + 1)**2 * corr_levels
        motion_encoder_input_dim = corr_feature_dim + 2 # Correlation features + flow (u,v)
        motion_encoder_output_dim = 32 # Chosen dimension for motion features
        inp_dim = max(0, context_dim - hidden_dim)
        gru_input_total_dim = motion_encoder_output_dim + inp_dim

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
        shape_tensor = tf.shape(net)
        b, h, w = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        if flow_init is None:
            flow = tf.zeros([b, h, w, 2], dtype=tf.float32)
        else:
            flow = tf.cast(flow_init, tf.float32)

        hidden_state = net
        flow_predictions = []
        for iter in range(self.iterations):
            flow = tf.stop_gradient(flow)
            motion_input = tf.concat([corr_features, flow], axis=-1)
            motion_features = self.motion_encoder(motion_input)
            gru_input = tf.concat([motion_features, inp], axis=-1)
            hidden_state, [hidden_state] = self.gru_cell(gru_input, [hidden_state])
            delta_flow = self.flow_head(hidden_state)
            delta_flow = tf.cast(delta_flow, tf.float32)
            flow = flow + delta_flow
            flow_predictions.append(flow)
        return flow_predictions

# Helper for Correlation
def build_correlation_volume(fmap1, fmap2, radius=4):
    compute_dtype = fmap1.dtype
    fmap2 = tf.cast(fmap2, compute_dtype)
    batch_size, h, w, c = tf.shape(fmap1)[0], tf.shape(fmap1)[1], tf.shape(fmap1)[2], tf.shape(fmap1)[3]
    pad_size = radius
    fmap2_padded = tf.pad(fmap2, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT')
    gy, gx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    coords_base = tf.stack([gy, gx], axis=-1)
    coords_base = tf.cast(coords_base, tf.int32)
    coords_base = tf.expand_dims(tf.expand_dims(coords_base, 0), -2)
    coords_base = tf.tile(coords_base, [batch_size, 1, 1, 1, 1])
    dy, dx = tf.meshgrid(tf.range(-radius, radius + 1), tf.range(-radius, radius + 1), indexing='ij')
    delta = tf.stack([dy, dx], axis=-1)
    num_neighbors = (2*radius+1)**2
    delta = tf.reshape(delta, [1, 1, 1, num_neighbors, 2])
    delta = tf.cast(delta, tf.int32)
    lookup_coords = coords_base + delta + pad_size
    batch_indices = tf.range(batch_size)
    batch_indices = tf.reshape(batch_indices, [batch_size, 1, 1, 1])
    batch_indices = tf.tile(batch_indices, [1, h, w, num_neighbors])
    lookup_indices = tf.stack([batch_indices, lookup_coords[..., 0], lookup_coords[..., 1]], axis=-1)
    fmap2_neighbors = tf.gather_nd(fmap2_padded, lookup_indices)
    fmap1_expanded = tf.expand_dims(fmap1, axis=3)
    correlation = tf.reduce_sum(fmap1_expanded * fmap2_neighbors, axis=-1)
    correlation_float32 = tf.cast(correlation, tf.float32)
    # Normalize correlation
    correlation_normalized = correlation_float32 / tf.maximum(tf.cast(c, tf.float32), 1e-6) # Avoid division by zero
    return correlation_normalized

# RAFT Model
class RAFT(tf.keras.Model):
    def __init__(self, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, num_iterations=NUM_ITERATIONS, hidden_dim=128, context_dim=128, corr_levels=1, corr_radius=4, name='raft', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.feature_encoder = FeatureEncoder()
        self.context_encoder = ContextEncoder()
        self.update_block = UpdateBlock(iterations=num_iterations, hidden_dim=hidden_dim, context_dim=context_dim, corr_levels=corr_levels, corr_radius=corr_radius)

    @tf.function
    def upsample_flow(self, flow, target_height, target_width):
        flow = tf.cast(flow, tf.float32)
        shape_tensor = tf.shape(flow)
        b, h_low, w_low = shape_tensor[0], shape_tensor[1], shape_tensor[2]
        # Ensure low dimensions are not zero before division
        h_low_safe = tf.maximum(tf.cast(h_low, tf.float32), 1.0)
        w_low_safe = tf.maximum(tf.cast(w_low, tf.float32), 1.0)

        scale_factor_h = tf.cast(target_height, tf.float32) / h_low_safe
        scale_factor_w = tf.cast(target_width, tf.float32) / w_low_safe

        flow_upsampled = tf.image.resize(flow, [target_height, target_width], method='bilinear')
        u = flow_upsampled[..., 0] * scale_factor_w
        v = flow_upsampled[..., 1] * scale_factor_h
        flow_scaled = tf.stack([u, v], axis=-1)
        return flow_scaled

    def call(self, inputs, training=False):
        image1, image2 = inputs
        target_height = tf.shape(image1)[1]
        target_width = tf.shape(image1)[2]

        # Ensure inputs are processed in compute dtype
        # Keras layers handle this internally based on the policy
        # image1 = tf.cast(image1, self.compute_dtype)
        # image2 = tf.cast(image2, self.compute_dtype)

        fmap1 = self.feature_encoder(image1)
        fmap2 = self.feature_encoder(image2)
        context_fmap = self.context_encoder(image1)

        split_sizes = [self.hidden_dim, max(0, self.context_dim - self.hidden_dim)]
        if sum(split_sizes) != self.context_dim:
             raise ValueError(f"Context split sizes {split_sizes} do not sum to context dimension {self.context_dim}")

        net, inp = tf.split(context_fmap, split_sizes, axis=-1)
        net = tf.tanh(net) # Initial hidden state for GRU
        inp = tf.nn.relu(inp) # Input features for GRU updates

        corr_features = build_correlation_volume(fmap1, fmap2, radius=self.corr_radius)
        # Correlation features should also be in compute dtype

        # Call update block
        flow_predictions_low_res = self.update_block(net, inp, corr_features, flow_init=None)

        # Upsample predictions (result is float32)
        flow_predictions_upsampled = [self.upsample_flow(flow_lr, target_height, target_width) for flow_lr in flow_predictions_low_res]
        return flow_predictions_upsampled

# --- [End of RAFT Model Definitions] ---


# --- 2. Dataset Loading (CORRECTED for new structure) ---
def parse_frame_num(filename):
    """Extracts frame number from filename like imX.png""" # Updated docstring
    # Handles potential paths before the filename
    basename = os.path.basename(filename)
    # Use regex to find 'im' followed by one or more digits, then '.png'
    match = re.search(r'im(\d+)\.png', basename) # <<< NEW PATTERN
    return int(match.group(1)) if match else -1

def load_and_preprocess_image(path):
    """Loads and preprocesses a single image."""
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
        # Normalize to [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Ensure shape is known for graph mode
        img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return img
    except Exception as e:
        tf.print(f"Error loading image {path}: {e}", output_stream=sys.stderr)
        # Return a dummy tensor of the correct shape and type to avoid dataset errors
        # Adjust dtype if not using float32
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

def configure_dataset_for_ae(data_dir, batch_size):
    """
    Configures dataset for Autoencoder training based on the corrected structure:
    data_dir/SCENE/SEQUENCE_ID/frame_XXXX.png
    Yields pairs of consecutive images. Includes debugging prints.
    """
    image_pairs = []
    print(f"Scanning dataset structure in: {data_dir}")
    print("-" * 30) # Separator

    # Iterate through scene folders (e.g., 00081, 00082...)
    scene_folders = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not scene_folders:
        raise ValueError(f"No scene folders found in {data_dir}. Check DATA_DIR.")
    print(f"Found {len(scene_folders)} potential scene folders.")
    # print(f"Example scene folders found: {scene_folders[:5]}...") # Uncomment to see examples

    total_sequences_found = 0
    total_frames_found = 0

    for scene_path in scene_folders:
        print(f"Processing scene: {os.path.basename(scene_path)}") # DEBUG
        # Iterate through sequence ID folders (e.g., 0001, 0002...)
        sequence_folders = sorted([os.path.join(scene_path, s) for s in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, s)) and s.isdigit()])
        # print(f"  Found sequence folders: {sequence_folders}") # DEBUG (can be very long)

        if not sequence_folders:
            # print(f"  No sequence subfolders found in {scene_path}") # DEBUG
            continue

        total_sequences_found += len(sequence_folders)

        for seq_path in sequence_folders:
            # Find all frame images matching the pattern within the sequence folder
            # Find all frame images matching the pattern within the sequence folder
            frame_pattern = os.path.join(seq_path, 'im*.png') # <<< NEW PATTERN
            frame_files = glob(frame_pattern)
            # print(f"    Processing sequence: {os.path.basename(seq_path)} - Checking pattern: {frame_pattern}") # DEBUG

            if not frame_files:
                # print(f"      No files found matching pattern '{frame_pattern}'") # DEBUG
                continue # Skip if glob finds nothing

            total_frames_found += len(frame_files)
            # print(f"      Found {len(frame_files)} files matching pattern.") # DEBUG

            # Sort frames numerically
            try:
                # Filter out frames where parse_frame_num returns -1 before sorting
                valid_frames = [(f, parse_frame_num(f)) for f in frame_files if parse_frame_num(f) != -1]
                if not valid_frames:
                    # print(f"      No valid frame numbers found in {seq_path}") # DEBUG
                    continue

                sorted_frames_with_nums = sorted(valid_frames, key=lambda item: item[1])
                sorted_frames = [item[0] for item in sorted_frames_with_nums] # Get only paths back
                # print(f"      Sorted valid frames ({len(sorted_frames)}): { [os.path.basename(f) for f in sorted_frames[:5]] }...") # DEBUG
            except Exception as e:
                 print(f"      Error sorting frames in {seq_path}: {e}. Skipping sequence.")
                 continue

            if len(sorted_frames) < 2:
                # print(f"      Less than 2 valid & sorted frames found.") # DEBUG
                continue

            # Create consecutive pairs within this sequence
            sequence_pairs_found = 0
            for i in range(len(sorted_frames) - 1):
                frame1_path = sorted_frames[i]
                frame2_path = sorted_frames[i+1]

                # Use the pre-parsed numbers
                frame1_num = sorted_frames_with_nums[i][1]
                frame2_num = sorted_frames_with_nums[i+1][1]
                # print(f"        Checking pair: {os.path.basename(frame1_path)} ({frame1_num}) and {os.path.basename(frame2_path)} ({frame2_num})") # DEBUG


                # Ensure they are truly consecutive
                if frame2_num == frame1_num + 1:
                    # print(f"          Found consecutive pair!") # DEBUG
                    # Check if files actually exist (robustness - glob should ensure this, but safe)
                    if os.path.exists(frame1_path) and os.path.exists(frame2_path):
                        image_pairs.append((frame1_path, frame2_path))
                        sequence_pairs_found += 1
                    # else:
                    #      print(f"          Warning: File missing for consecutive pair {frame1_path}, {frame2_path}")
                # else:
                #     print(f"          Not consecutive.") # DEBUG
            # if sequence_pairs_found > 0:
            #      print(f"      Found {sequence_pairs_found} pairs in sequence {os.path.basename(seq_path)}")

    print("-" * 30) # Separator
    print(f"Scan Summary:")
    print(f"  Total sequences processed (attempted): {total_sequences_found}")
    print(f"  Total frame files found matching 'frame_*.png': {total_frames_found}")
    print(f"  Total consecutive image pairs found: {len(image_pairs)}")
    print("-" * 30) # Separator


    if not image_pairs:
        print("\n!!! ERROR DETAILS !!!")
        print("No valid consecutive image pairs were found based on the scanning.")
        print("Potential issues to check:")
        print("  1. Correct `DATA_DIR`? Is it pointing to the directory containing '00081', '00082', etc.?")
        print(f"     Current DATA_DIR: '{data_dir}'")
        print("  2. Correct Subfolder Structure? Does `DATA_DIR/SCENE_ID/SEQUENCE_ID/` exist?")
        print("  3. Correct Frame Filename Pattern? Are frames named EXACTLY `frame_XXXX.png` (e.g., frame_0001.png, frame_0002.png)?")
        print("     - Check file extensions (.png vs .jpg, etc.)")
        print("     - Check the prefix ('frame_' vs something else)")
        print("     - Check the numbering format (zero-padding?)")
        print("  4. Are there actually *consecutive* frames (like frame_0001.png AND frame_0002.png) present in the sequence folders?")
        raise ValueError("No valid consecutive image pairs found. Check dataset structure and frame naming (frame_XXXX.png).")

    # --- Rest of the function ---
    print(f"Successfully found {len(image_pairs)} consecutive image pairs.")
    # Create datasets of image paths
    img_path1_ds = tf.data.Dataset.from_tensor_slices([p[0] for p in image_pairs])
    img_path2_ds = tf.data.Dataset.from_tensor_slices([p[1] for p in image_pairs])

    # Map paths to loaded & preprocessed images
    image1_ds = img_path1_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image2_ds = img_path2_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Zip the two image datasets together
    dataset = tf.data.Dataset.zip((image1_ds, image2_ds))

    # Shuffle the pairs for training randomness
    dataset = dataset.shuffle(buffer_size=min(len(image_pairs), 500))
    dataset = dataset.batch(batch_size)
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# --- 3. Flow Autoencoder Model Implementation ---
# Vector Quantizer Layer
class VectorQuantizer(tf.keras.layers.Layer):
    """
    Vector Quantization layer implementing the Straight-Through Estimator (STE).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the codebook/embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(embedding_dim, num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vq",
        )

    def call(self, inputs):
        # inputs shape: (batch, height, width, embedding_dim), dtype=compute_dtype (e.g., float16)
        input_dtype = inputs.dtype # Store original dtype (e.g., float16)
        inputs_float32 = tf.cast(inputs, dtype=tf.float32) # Cast to float32 for calculations

        flat_inputs = tf.reshape(inputs_float32, [-1, self.embedding_dim])

        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat_inputs, self.embeddings)
            + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True)
        )

        encoding_indices = tf.argmin(distances, axis=1)
        quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.embeddings), encoding_indices) # float32

        input_shape = tf.shape(inputs)
        quantized_spatial = tf.reshape(quantized_flat, input_shape) # float32

        # --- Calculate VQ Loss (using float32) ---
        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized_spatial) - inputs_float32))
        q_latent_loss = tf.reduce_mean(tf.square(quantized_spatial - tf.stop_gradient(inputs_float32)))
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        self.add_loss(tf.cast(loss, dtype=tf.float32))

        # --- Straight-Through Estimator (STE) ---
        # Calculate the difference for STE using float32 for both operands
        difference_f32 = quantized_spatial - inputs_float32 # Both are float32

        # Apply stop_gradient to the float32 difference
        stopped_difference_f32 = tf.stop_gradient(difference_f32)

        # --- *** FIX: Cast the difference back to the original input dtype before adding *** ---
        stopped_difference_casted = tf.cast(stopped_difference_f32, input_dtype)

        # Add the casted difference back to the original 'inputs' tensor.
        # Both operands are now the same dtype (e.g., float16).
        quantized_ste = inputs + stopped_difference_casted

        # Return the quantized vectors (with STE for backprop) and the indices
        return quantized_ste, encoding_indices # quantized_ste should be compute_dtype (e.g., float16)
# Autoencoder Architecture
def build_encoder(input_shape, latent_dim):
    # Input shape is (H, W, 2) for flow
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32) # Expect float32 flow maps
    # Cast to compute_dtype if using mixed precision
    x = tf.cast(inputs, tf.keras.mixed_precision.global_policy().compute_dtype)

    # Example Encoder layers (adjust complexity as needed)
    # Using LeakyReLU might be slightly better for VAEs/AAs than ReLU
    activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)

    x = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(x) # H/2, W/2
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x) # Use float32 for BN stability
    x = activation_fn(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x) # H/4, W/4
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x) # H/8, W/8
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x) # Keep dims
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)

    # Final conv layer to match the VQ embedding dimension
    # No activation here, the VQ layer takes the raw embedding vector
    latent = tf.keras.layers.Conv2D(latent_dim, kernel_size=1, padding='same', name='encoder_output')(x) # H/8, W/8, latent_dim
    # Output dtype will be compute_dtype (e.g., float16)

    return tf.keras.Model(inputs, latent, name='flow_encoder')

def build_decoder(latent_shape, latent_dim):
    # Input shape is the shape of the quantized latent space (H', W', latent_dim)
    latent_inputs = tf.keras.Input(shape=latent_shape, dtype=tf.keras.mixed_precision.global_policy().compute_dtype)
    x = latent_inputs
    activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)

    # Use Conv2DTranspose to upsample
    # Ensure Conv2DTranspose output dtype matches compute_dtype if needed
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(x) # H/4, W/4
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x) # H/2, W/2
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x) # H, W
    x = tf.keras.layers.BatchNormalization(dtype=tf.float32)(x)
    x = activation_fn(x)


    # Final layer to reconstruct the 2-channel flow map
    # Use linear activation for flow values.
    # Output should be float32 for precise flow values.
    outputs = tf.keras.layers.Conv2DTranspose(2, kernel_size=3, padding='same', activation='linear', dtype='float32', name='decoder_output')(x) # H, W, 2

    return tf.keras.Model(latent_inputs, outputs, name='flow_decoder')

class FlowVQAutoencoder(tf.keras.Model):
    def __init__(self, input_shape, latent_dim, num_embeddings, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.input_flow_shape = input_shape
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.encoder = build_encoder(input_shape, latent_dim)

        # Determine the shape of the latent space after encoding
        # Use float32 input for shape determination
        dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
        latent_output = self.encoder(dummy_input)
        # Extract spatial dimensions (H', W') from the symbolic shape tensor
        self.latent_spatial_shape = tf.shape(latent_output)[1:3].numpy()
        print(f"Autoencoder Determined Latent Spatial Shape: {self.latent_spatial_shape}")
        # Full latent shape including channels
        self.latent_shape = (self.latent_spatial_shape[0], self.latent_spatial_shape[1], latent_dim)

        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, name="vector_quantizer")
        self.decoder = build_decoder(self.latent_shape, latent_dim)

        # Metrics to track losses
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.vq_loss_tracker = tf.keras.metrics.Mean(name="vq_loss")

    @property
    def metrics(self):
        # List the metrics to be tracked and reset at each epoch/step
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.vq_loss_tracker,
        ]

    def call(self, inputs, training=False):
        # Input `inputs` is the original flow map (B, H, W, 2), expect float32
        encoded = self.encoder(inputs, training=training)
        # encoded shape: (B, H', W', latent_dim), dtype=compute_dtype

        quantized_ste, indices = self.vq_layer(encoded) # VQ handles internal types and STE
        # quantized_ste shape: (B, H', W', latent_dim), dtype=compute_dtype
        # indices shape: (B*H'*W',)

        reconstructed = self.decoder(quantized_ste, training=training)
        # reconstructed shape: (B, H, W, 2), dtype=float32

        return reconstructed, indices # Return reconstruction and the discrete indices

    def train_step(self, data):
        """ Overrides train_step for custom training logic with dynamic flow generation. """
        image_pair = data # Dataset yields image pairs

        # 1. Generate 'ground truth' flow using frozen RAFT model
        image1, image2 = image_pair
        # Ensure RAFT runs on GPU if available and in inference mode
        # RAFT output is a list, take the last one (most refined)
        # RAFT output is float32
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
             flow_predictions_raft = raft_model_generator([image1, image2], training=False)
        flow_target = flow_predictions_raft[-1] # Shape (B, H, W, 2), float32
        flow_target = tf.stop_gradient(flow_target) # Ensure no gradients flow back to RAFT

        # 2. Train the Autoencoder
        with tf.GradientTape() as tape:
            reconstructed_flow, _ = self(flow_target, training=True) # Use self() to call the model

            # Calculate reconstruction loss (ensure inputs are float32)
            recon_loss = reconstruction_loss_fn(flow_target, reconstructed_flow) # MSE

            # Get VQ loss (added internally in the VQ layer, sum them up)
            # VQ loss is already float32
            vq_loss = sum(self.vq_layer.losses)

            # Combine losses (both should be float32)
            total_loss = recon_loss + vq_loss

            # Scale loss for mixed precision if optimizer requires it
            scaled_loss = total_loss
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        # Compute and apply gradients for the Autoencoder
        gradients = tape.gradient(scaled_loss, self.trainable_variables)

        # Unscale gradients if using mixed precision LossScaleOptimizer
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # Clip gradients (optional but often helpful)
        gradients = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in gradients]

        # Apply gradients
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None]
        self.optimizer.apply_gradients(valid_grads_and_vars)

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)

        # Return loss dictionary for Keras progress bar/logging
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """ Defines evaluation step (optional but good practice). """
        image_pair = data
        image1, image2 = image_pair

        # Generate flow target
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            flow_predictions_raft = raft_model_generator([image1, image2], training=False)
        flow_target = flow_predictions_raft[-1]
        flow_target = tf.stop_gradient(flow_target)

        # Get reconstruction
        reconstructed_flow, _ = self(flow_target, training=False)

        # Calculate losses
        recon_loss = reconstruction_loss_fn(flow_target, reconstructed_flow)
        vq_loss = sum(self.vq_layer.losses) # Note: VQ loss might behave differently in eval if stateful
        total_loss = recon_loss + vq_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)

        return {m.name: m.result() for m in self.metrics}


    # Helper methods for inference/analysis
    def get_vq_loss(self):
        """ Returns the summed VQ loss computed within the VQ layer. """
        # Ensure losses list is not empty and sum correctly
        return sum(self.vq_layer.losses) if self.vq_layer.losses else tf.constant(0.0, dtype=tf.float32)

    def encode_to_indices(self, flow_input):
        """ Encodes flow input to discrete latent indices. """
        # Ensure input is float32
        flow_input_f32 = tf.cast(flow_input, tf.float32)
        encoded = self.encoder(flow_input_f32, training=False)
        # Get indices from VQ layer (does not use STE path)
        # Need to replicate VQ logic slightly without STE/loss calculation
        encoded_f32 = tf.cast(encoded, tf.float32)
        flat_encoded = tf.reshape(encoded_f32, [-1, self.latent_dim])
        distances = (
            tf.reduce_sum(flat_encoded**2, axis=1, keepdims=True)
            - 2 * tf.matmul(flat_encoded, self.vq_layer.embeddings)
            + tf.reduce_sum(self.vq_layer.embeddings**2, axis=0, keepdims=True)
        )
        indices = tf.argmin(distances, axis=1) # Shape: (B*H'*W',)

        # Reshape indices back to spatial form B, H', W'
        batch_size = tf.shape(flow_input)[0]
        indices_reshaped = tf.reshape(indices, (batch_size, self.latent_spatial_shape[0], self.latent_spatial_shape[1]))
        return indices_reshaped

    def decode_from_indices(self, indices):
        """ Decodes flow from discrete latent indices. """
        # indices shape: (batch, H', W')
        # Lookup embeddings (codebook vectors) using indices
        flat_indices = tf.reshape(indices, [-1])
        # Embeddings are float32
        quantized_flat = tf.nn.embedding_lookup(tf.transpose(self.vq_layer.embeddings), flat_indices)
        # Reshape back to spatial dimensions B, H', W', C
        quantized_spatial = tf.reshape(quantized_flat,
                                       (tf.shape(indices)[0], self.latent_spatial_shape[0], self.latent_spatial_shape[1], self.latent_dim))
        # Cast to compute dtype for decoder input
        quantized_compute = tf.cast(quantized_spatial, self.compute_dtype)
        # Decode
        reconstructed_flow = self.decoder(quantized_compute, training=False)
        return reconstructed_flow # float32 output

# --- 4. Training Setup ---

# Load and prepare the pre-trained RAFT model
print("Loading pre-trained RAFT model for flow generation...")
raft_model_generator = RAFT(num_iterations=NUM_ITERATIONS)
# Build RAFT model by calling it once (needed before loading weights)
dummy_img1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
dummy_img2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
try:
    # Call with specific shapes to ensure layers are built correctly
    _ = raft_model_generator([dummy_img1, dummy_img2], training=False)
    print("RAFT model built.")
    if os.path.exists(RAFT_CHECKPOINT_PATH):
        print(f"Attempting to load RAFT weights from {RAFT_CHECKPOINT_PATH}...")
        # Load the weights. This will raise an error if incompatible or file is bad.
        raft_model_generator.load_weights(RAFT_CHECKPOINT_PATH)
        # If the above line succeeds without error, print the success message.
        # We can often omit expect_partial() when just loading weights for inference,
        # as long as the core layers match.
        print(f"Successfully loaded RAFT weights from {RAFT_CHECKPOINT_PATH}")
    else:
        # If the file doesn't exist
        print(f"!!! ERROR: RAFT checkpoint file not found at {RAFT_CHECKPOINT_PATH}. !!!")
        print("!!! Cannot proceed without the pre-trained RAFT weights. !!!")
        raise FileNotFoundError(f"RAFT checkpoint not found: {RAFT_CHECKPOINT_PATH}")

except Exception as e:
    # Catch any exception during building or loading
    print(f"Error building or loading RAFT model: {e}")
    print(f"Please check:")
    print(f"  1. If the RAFT checkpoint path is correct: {RAFT_CHECKPOINT_PATH}")
    print(f"  2. If the RAFT model definition in this script EXACTLY matches the one used for training.")
    print(f"  3. If the checkpoint file is corrupted.")
    raise SystemExit("Cannot proceed without a loadable RAFT model.")
# Freeze RAFT model weights - crucial!
raft_model_generator.trainable = False
print("RAFT model frozen for flow generation.")


# Instantiate the Autoencoder
print("Instantiating Flow VQ Autoencoder...")
flow_ae = FlowVQAutoencoder(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 2),
    latent_dim=LATENT_DIM,
    num_embeddings=NUM_EMBEDDINGS,
    commitment_cost=COMMITMENT_COST
)

# Optimizer for Autoencoder
print("Configuring Optimizer...")
ae_optimizer = tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE)
# LossScaleOptimizer is handled implicitly by Keras >= 2.4 when policy is set
# No need to explicitly wrap the optimizer anymore if using model.compile

# Compile the Autoencoder model
# Using model.compile simplifies training loop and metric handling
print("Compiling Autoencoder model...")
flow_ae.compile(optimizer=ae_optimizer)
# Build the model by processing a dummy input AFTER compiling
dummy_flow = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 2], dtype=tf.float32)
_ = flow_ae(dummy_flow) # Call the model to build layers
print("Flow Autoencoder built and compiled.")
flow_ae.summary() # Print model summary


# Loss Function for Autoencoder (Reconstruction) - Defined for clarity, used in train_step
reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

# Checkpoint Manager for Autoencoder
# We need to track the model and optimizer state. Epoch/step managed by Keras callbacks/fit.
ae_ckpt = tf.train.Checkpoint(model=flow_ae, optimizer=ae_optimizer)
ae_ckpt_manager = tf.train.CheckpointManager(ae_ckpt, AE_CHECKPOINT_DIR, max_to_keep=5)

# Restore Autoencoder Checkpoint (before starting training)
initial_epoch = 0
if ae_ckpt_manager.latest_checkpoint:
    print(f"Restoring AE checkpoint from {ae_ckpt_manager.latest_checkpoint}...")
    # Restore model weights and optimizer state
    status = ae_ckpt.restore(ae_ckpt_manager.latest_checkpoint)
    try:
        # Check if restoration was successful (might need expect_partial if optimizer state changed)
        status.assert_existing_objects_matched().expect_partial()
        print("AE Checkpoint restored successfully.")
        # Extract epoch number from checkpoint filename if possible (e.g., ckpt-10)
        try:
            initial_epoch = int(ae_ckpt_manager.latest_checkpoint.split('-')[-1])
            print(f"Resuming AE training from Epoch {initial_epoch}")
        except:
            print("Could not determine resume epoch from checkpoint filename. Starting from epoch 0.")
            initial_epoch = 0 # Fallback
    except AssertionError as e:
        print(f"Warning: AE Checkpoint restoration issue: {e}. Training from scratch.")
        initial_epoch = 0
else:
    print("No AE checkpoint found, initializing AE from scratch.")
    initial_epoch = 0


# --- 5. Training Loop using model.fit ---
print(f"Preparing Autoencoder training dataset...")
ae_train_dataset = configure_dataset_for_ae(DATA_DIR, batch_size=AE_BATCH_SIZE)

# Calculate steps per epoch if dataset size is known
steps_per_epoch = None
ae_total_batches_per_epoch = tf.data.experimental.cardinality(ae_train_dataset)
if ae_total_batches_per_epoch != tf.data.experimental.UNKNOWN_CARDINALITY:
     steps_per_epoch = ae_total_batches_per_epoch.numpy()
     print(f"Dataset size determined: {steps_per_epoch} steps per epoch.")
else:
     print("Could not determine dataset size. Steps per epoch unknown.")


# TensorBoard Callback Setup for Autoencoder
ae_current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ae_log_dir_epoch = os.path.join(AE_LOG_DIR, ae_current_time + f"_start_epoch_{initial_epoch}")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=ae_log_dir_epoch,
    histogram_freq=1, # Log histograms every epoch (can be memory intensive)
    profile_batch=0 # Disable profiler by default (can enable: '10,20')
)
print(f"Logging AE TensorBoard data to: {ae_log_dir_epoch}")

# Checkpoint Callback Setup
# Saves the model weights after each epoch (or based on save_freq if steps_per_epoch is known)
# Filename includes epoch number. Using tf.train.Checkpoint format for flexibility.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(AE_CHECKPOINT_DIR, 'ckpt-{epoch:04d}'), # Standard Keras format
    save_weights_only=True, # Save only weights (compatible with tf.train.Checkpoint restore)
    save_freq='epoch', # Save every epoch. Could use integer steps if steps_per_epoch is known.
    verbose=1
)

# Checkpoint saving using tf.train.CheckpointManager within a callback
class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        # Save checkpoint at the end of each epoch
        save_path = self.checkpoint_manager.save(checkpoint_number=epoch + 1)
        print(f"\nEpoch {epoch+1}: Saved TF Checkpoint to {save_path}")

# Use the CheckpointManager callback
checkpoint_manager_callback = CheckpointManagerCallback(ae_ckpt_manager)


print(f"\n--- Starting Autoencoder Training ---")
print(f"Training from epoch {initial_epoch} up to {AE_EPOCHS}...")

history = flow_ae.fit(
    ae_train_dataset,
    epochs=AE_EPOCHS,
    initial_epoch=initial_epoch, # Start from the restored epoch
    callbacks=[
        tensorboard_callback,
        checkpoint_manager_callback # Use our custom manager callback
        # checkpoint_callback # Or use the standard Keras callback
    ],
    steps_per_epoch=steps_per_epoch # Only needed if dataset size is unknown and you want a defined epoch length
)

print("\nAutoencoder Training finished.")

# --- 6. Model Saving (Autoencoder Weights) ---
# The CheckpointManagerCallback already saves checkpoints.
# Optionally, save the final weights explicitly.
print("Saving final Autoencoder weights...")
final_ae_weights_path = os.path.join(AE_CHECKPOINT_DIR, 'flow_ae_final_weights.weights.h5')
try:
    flow_ae.save_weights(final_ae_weights_path)
    print(f"Final AE weights saved to {final_ae_weights_path}")
except Exception as e:
    print(f"Error saving final AE weights: {e}")


# --- 7. Inference and Visualization (Autoencoder Reconstruction) ---
print("\nRunning Autoencoder inference and visualization...")

# Reuse visualization functions (make_color_wheel, flow_to_color, visualize_flow)
# --- [Start of Visualization Helpers] ---
def make_color_wheel():
    """Generates a color wheel for flow visualization."""
    RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255; colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY)/RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0, YG)/YG); colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255; colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0, GC)/GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(0, CB)/CB); colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255; colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0, BM)/BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(0, MR)/MR); colorwheel[col:col+MR, 0] = 255
    return colorwheel.astype(np.uint8)

def flow_to_color(flow, convert_to_bgr=False):
    """Converts optical flow (u, v) to a color image using Middlebury color scheme."""
    # Input flow: (H, W, 2), numpy array, float32
    if flow is None:
        # Return a default size black image if flow is None
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    UNKNOWN_FLOW_THRESH = 1e7
    SMALL_FLOW = 1e-9 # Avoid division by zero

    # Check if flow has expected dimensions, handle case where it might be invalid shape
    if not isinstance(flow, np.ndarray) or flow.ndim != 3 or flow.shape[2] != 2:
        print(f"Warning: Invalid flow shape received in flow_to_color: {flow.shape if isinstance(flow, np.ndarray) else type(flow)}. Returning black image.")
        # Use configured IMG_HEIGHT, IMG_WIDTH for default size
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    height, width, _ = flow.shape
    if height == 0 or width == 0:
        print(f"Warning: Flow has zero dimension in flow_to_color: {flow.shape}. Returning black image.")
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    colorwheel = make_color_wheel()
    ncols = colorwheel.shape[0]

    # Separate u and v components
    u, v = flow[..., 0], flow[..., 1]

    # Handle NaNs or Infs if any (replace with 0)
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)

    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(-v, -u) / np.pi # Range -1 to 1

    # Normalize angle to 0-1 range
    ang = (ang + 1.0) / 2.0

    # --- Robust magnitude normalization ---
    # Consider only valid, finite magnitudes for normalization max
    valid_indices = np.isfinite(mag) & (np.abs(mag) < UNKNOWN_FLOW_THRESH)
    if np.any(valid_indices):
        mag_max = np.max(mag[valid_indices])
        if mag_max <= SMALL_FLOW: # Handle case where max magnitude is very small or zero
             mag_max = SMALL_FLOW # Avoid division by zero or near-zero issues
    else:
        mag_max = SMALL_FLOW # If no valid magnitudes, use a small default

    # Normalize magnitude 0-1, clipping potential outliers slightly above mag_max
    mag_norm = np.clip(mag / mag_max, 0, 1)
    # --- End robust normalization ---


    # Map angle to color wheel index
    fk = (ang * (ncols - 1)) # Map 0-1 angle to 0-(ncols-1) index
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0 # Interpolation factor 0-1

    # Interpolate colors
    for i in range(colorwheel.shape[1]): # R, G, B
        tmp = colorwheel[:, i]
        # Ensure indices are within bounds (redundant due to modulo, but safe)
        k0 = np.clip(k0, 0, ncols - 1)
        k1 = np.clip(k1, 0, ncols - 1)

        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        # Increase saturation with magnitude (desaturate towards white)
        col = 1 - mag_norm * (1 - col)
        # Ensure valid pixel values [0, 255]
        img[:, :, i] = np.clip(np.floor(255.0 * col), 0, 255)

    # Handle unknown flow (mark as black)
    # Use the original non-normalized magnitude for threshold check
    idx_unknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | \
                  (np.abs(v) > UNKNOWN_FLOW_THRESH) | \
                  ~valid_indices # Also mark NaNs/Infs found earlier as unknown
    img[idx_unknown] = 0

    if convert_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def visualize_flow(image1_np, flow_pred_np, filename_prefix="flow_vis"):
    """Generates and saves flow visualizations."""
    if image1_np is None or flow_pred_np is None:
        print("Warning: Cannot visualize flow, input image or flow is None.")
        return
    # Ensure inputs are numpy arrays
    image1_np = np.asarray(image1_np)
    flow_pred_np = np.asarray(flow_pred_np)

    # Clamp image display range to [0, 1] if it's float
    if image1_np.dtype == np.float32 or image1_np.dtype == np.float64:
        image1_np = np.clip(image1_np, 0, 1)
    # Convert to uint8 if needed by imshow
    if image1_np.dtype != np.uint8:
         image1_np_display = (image1_np * 255).astype(np.uint8)
    else:
         image1_np_display = image1_np


    # Ensure flow_pred_np has correct shape
    if flow_pred_np.ndim != 3 or flow_pred_np.shape[-1] != 2:
        print(f"Warning: Invalid flow shape for visualization: {flow_pred_np.shape}. Skipping visualization.")
        return
    h, w, _ = flow_pred_np.shape # Use flow shape for consistency
    img_h, img_w, _ = image1_np_display.shape

    # Resize image to match flow if necessary
    if img_h != h or img_w != w:
        print(f"Warning: Image shape ({img_h}x{img_w}) doesn't match flow shape ({h}x{w}). Resizing image for display.")
        image1_np_display = cv2.resize(image1_np_display, (w, h), interpolation=cv2.INTER_LINEAR) # cv2 uses (width, height)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filename_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Flow Magnitude Heatmap
    try:
        plt.figure(figsize=(10, 8))
        # Use only finite values for magnitude calculation
        u_finite, v_finite = np.nan_to_num(flow_pred_np[..., 0]), np.nan_to_num(flow_pred_np[..., 1])
        magnitude = np.sqrt(u_finite**2 + v_finite**2)
        # Determine color limits based on finite values, excluding extreme outliers maybe
        mag_finite = magnitude[np.isfinite(magnitude)]
        vmin = np.percentile(mag_finite, 1) if mag_finite.size > 0 else 0
        vmax = np.percentile(mag_finite, 99) if mag_finite.size > 0 else 1
        im = plt.imshow(magnitude, cmap='viridis', vmin=vmin, vmax=vmax) # Use viridis or jet
        plt.colorbar(im, label='Flow Magnitude (pixels)')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Magnitude')
        plt.axis('off')
        plt.tight_layout()
        save_path = f"{filename_prefix}_magnitude.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved flow magnitude heatmap to {save_path}")
    except Exception as e:
        print(f"Error generating magnitude plot for {filename_prefix}: {e}")
        plt.close() # Ensure plot is closed on error

    # 2. Vector Field Overlay (Quiver Plot)
    try:
        step = max(1, min(h, w) // 32) # Adjust density based on image size
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        # Get flow vectors at sampled points, handle potential NaNs
        fx, fy = np.nan_to_num(flow_pred_np[y, x].T)

        plt.figure(figsize=(12, 9))
        plt.imshow(image1_np_display) # Show image 1 as background
        # Adjust quiver scale: smaller values -> longer arrows. May need tuning.
        # Use scale_units='xy' so scale is relative to plot axes. Angles='xy' ensures correct direction.
        plt.quiver(x, y, fx, fy, color='red', scale=None, scale_units='xy', angles='xy',
                   headwidth=5, headlength=6, width=0.0015, pivot='tail')
        plt.title(f'{os.path.basename(filename_prefix)} - Flow Vectors (Overlay)')
        plt.axis('off')
        plt.tight_layout()
        save_path = f"{filename_prefix}_vectors.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved flow vector field to {save_path}")
    except Exception as e:
        print(f"Error generating vector plot for {filename_prefix}: {e}")
        plt.close() # Ensure plot is closed on error

    # 3. Middlebury Color Visualization
    try:
        flow_color_img = flow_to_color(flow_pred_np) # Use the dedicated function
        plt.figure(figsize=(10, 8))
        plt.imshow(flow_color_img)
        plt.title(f'{os.path.basename(filename_prefix)} - Flow (Color)')
        plt.axis('off')
        plt.tight_layout()
        save_path = f"{filename_prefix}_color.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved color flow visualization to {save_path}")
    except Exception as e:
        print(f"Error generating color flow plot for {filename_prefix}: {e}")
        plt.close() # Ensure plot is closed on error

# --- Run Inference and Visualization ---
print("\nRunning Autoencoder inference and visualization...")
vis_output_dir = "./visualization_output"
os.makedirs(vis_output_dir, exist_ok=True)

# Construct path to the visualization sequence frames
vis_sequence_path = os.path.join(DATA_DIR, VISUALIZATION_SCENE, VISUALIZATION_SEQUENCE)
scene_frames = sorted(glob(os.path.join(vis_sequence_path, 'frame_*.png')), key=parse_frame_num)

if len(scene_frames) < 2:
    print(f"Error: Need at least two consecutive frames in '{vis_sequence_path}' for AE visualization.")
    print(f"Please check VISUALIZATION_SCENE ('{VISUALIZATION_SCENE}') and VISUALIZATION_SEQUENCE ('{VISUALIZATION_SEQUENCE}').")
else:
    # Select first two consecutive frames found
    frame1_path = scene_frames[0]
    frame2_path = scene_frames[1]
    frame1_num = parse_frame_num(frame1_path)
    frame2_num = parse_frame_num(frame2_path)

    if frame2_num != frame1_num + 1:
         print(f"Warning: First two frames found ({os.path.basename(frame1_path)}, {os.path.basename(frame2_path)}) are not consecutive. Visualization might be misleading.")
         # Optionally: search for the first consecutive pair
         found_pair = False
         for i in range(len(scene_frames) - 1):
             f1_p = scene_frames[i]
             f2_p = scene_frames[i+1]
             f1_n = parse_frame_num(f1_p)
             f2_n = parse_frame_num(f2_p)
             if f1_n != -1 and f2_n == f1_n + 1:
                 frame1_path = f1_p
                 frame2_path = f2_p
                 frame1_num = f1_n
                 print(f"Using consecutive pair: {os.path.basename(frame1_path)} and {os.path.basename(frame2_path)}")
                 found_pair = True
                 break
         if not found_pair:
             print(f"Error: No consecutive frame pair found in {vis_sequence_path}. Cannot visualize.")
             exit() # Or handle differently


    print(f"Visualizing AE reconstruction for flow between: {os.path.basename(frame1_path)} and {os.path.basename(frame2_path)}")

    # Load images
    img1_inf = load_and_preprocess_image(frame1_path)
    img2_inf = load_and_preprocess_image(frame2_path)
    # Add batch dimension
    img1_inf_batch = tf.expand_dims(img1_inf, 0)
    img2_inf_batch = tf.expand_dims(img2_inf, 0)

    # 1. Generate the original flow map using RAFT (ensure model is in eval mode)
    print("Generating original flow map with RAFT...")
    start_raft_inf = time.time()
    # Use @tf.function(experimental_relax_shapes=True) or ensure fixed shapes if errors occur
    original_flow_list = raft_model_generator([img1_inf_batch, img2_inf_batch], training=False)
    raft_inf_time = time.time() - start_raft_inf
    original_flow_tf = original_flow_list[-1] # Use final prediction (B, H, W, 2), float32
    original_flow_np = original_flow_tf[0].numpy() # Remove batch dim, convert to numpy
    print(f"RAFT inference time: {raft_inf_time:.3f}s")

    # 2. Pass the original flow through the trained Autoencoder
    print("Compressing and reconstructing flow with Autoencoder...")
    start_ae_inf = time.time()
    reconstructed_flow_tf, _ = flow_ae(original_flow_tf, training=False) # Pass batch tensor (float32)
    ae_inf_time = time.time() - start_ae_inf
    reconstructed_flow_np = reconstructed_flow_tf[0].numpy() # Remove batch dim, convert to numpy (float32)
    print(f"Autoencoder inference time: {ae_inf_time:.3f}s")

    # 3. Visualize both original and reconstructed flows
    img1_np_vis = img1_inf.numpy() # Get numpy image for visualization background

    # Visualize Original RAFT Flow
    vis_filename_prefix_orig = os.path.join(vis_output_dir, f"vis_ae_{VISUALIZATION_SCENE}_{VISUALIZATION_SEQUENCE}_frame{frame1_num:04d}_ORIGINAL")
    print("\nVisualizing ORIGINAL flow (RAFT output):")
    visualize_flow(img1_np_vis, original_flow_np, filename_prefix=vis_filename_prefix_orig)

    # Visualize Reconstructed Flow
    vis_filename_prefix_recon = os.path.join(vis_output_dir, f"vis_ae_{VISUALIZATION_SCENE}_{VISUALIZATION_SEQUENCE}_frame{frame1_num:04d}_RECONSTRUCTED")
    print("\nVisualizing RECONSTRUCTED flow (AE output):")
    visualize_flow(img1_np_vis, reconstructed_flow_np, filename_prefix=vis_filename_prefix_recon)

    # 4. Calculate and print reconstruction EPE / MSE
    # EPE (End-Point Error)
    epe_map = tf.sqrt(tf.reduce_sum(tf.square(original_flow_tf - reconstructed_flow_tf), axis=-1) + 1e-8)
    epe = tf.reduce_mean(epe_map).numpy()
    # MSE (Mean Squared Error) - loss function used
    mse = reconstruction_loss_fn(original_flow_tf, reconstructed_flow_tf).numpy()
    print(f"\nReconstruction Quality on this sample:")
    print(f"  - EPE (End-Point Error): {epe:.4f} pixels")
    print(f"  - MSE (Mean Squared Error): {mse:.6f}")


    # 5. Simulate bitstream generation and theoretical size
    print("\nSimulating bitstream generation:")
    indices = flow_ae.encode_to_indices(original_flow_tf) # Get indices (B, H', W')
    indices_np = indices[0].numpy() # Remove batch dim
    unique_indices, counts = np.unique(indices_np, return_counts=True)
    print(f"  - Encoded flow to latent indices of shape: {indices_np.shape}")
    print(f"  - Number of unique codebook indices used: {len(unique_indices)} / {flow_ae.num_embeddings}")

    # Calculate theoretical entropy (lower bound for compression)
    probabilities = counts / indices_np.size
    entropy = -np.sum(probabilities * np.log2(probabilities))
    theoretical_min_bits = entropy * indices_np.size

    # Calculate raw bit cost (fixed length coding)
    bits_per_index = np.ceil(np.log2(flow_ae.num_embeddings)) # Use ceil for bits needed
    raw_bits = indices_np.size * bits_per_index

    print(f"  - Bits per index (fixed length code): {bits_per_index:.2f} bits")
    print(f"  - Theoretical raw bit cost (fixed): {raw_bits:.0f} bits (~{raw_bits/8/1024:.2f} KB)")
    print(f"  - Average bits per index (entropy): {entropy:.3f} bits")
    print(f"  - Theoretical minimum bit cost (entropy): {theoretical_min_bits:.0f} bits (~{theoretical_min_bits/8/1024:.2f} KB)")
    print(f"  - (Actual size depends on entropy coder efficiency, e.g., Huffman, Arithmetic)")

    # Example: Decode from indices to verify
    # reconstructed_from_indices_tf = flow_ae.decode_from_indices(indices)
    # reconstructed_from_indices_np = reconstructed_from_indices_tf[0].numpy()
    # Verify EPE between original reconstruction and decoding from indices (should be near zero)
    # epe_decode_check = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(reconstructed_flow_tf - reconstructed_from_indices_tf), axis=-1) + 1e-8)).numpy()
    # print(f"  - EPE check (Recon vs Decode from Indices): {epe_decode_check:.6f}")


print("\nFlow Autoencoder Script Complete.")