# hyperprior.py

import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
import sys

# --- MaskedConv2D (Keep as before) ---
class MaskedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, mask_type='A', **kwargs):
        super().__init__(**kwargs)
        if mask_type not in {'A'}:
            raise ValueError(f"Invalid mask type {mask_type}, only 'A' is supported.")
        self.mask_type = mask_type
        self._mask_internal = None

    def build(self, input_shape):
        super().build(input_shape)
        if not self.built:
            raise ValueError("Build should be called after the layer's kernel is created.")
        kernel_shape = tf.shape(self.kernel).numpy()
        h, w = self.kernel_size[0], self.kernel_size[1]
        mid_h, mid_w = h // 2, w // 2
        mask = np.zeros(kernel_shape, dtype=np.float32)
        mask[:mid_h, :, :, :] = 1.0
        mask[mid_h, :mid_w, :, :] = 1.0
        self._mask_internal = tf.constant(mask, dtype=tf.float32)

    def call(self, inputs):
        if self._mask_internal is None:
             raise RuntimeError("MaskedConv2D mask was not created during build.")
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

# --- Hyperprior + Entropy Coding Class ---
class HyperpriorSC(tf.keras.Model):
    """
    Hyperprior Network with Spatial Context (SC) modeling for VQ indices.
    Uses tfc.EntropyBottleneck for hyper-latent z.
    Uses tfc.CategoricalEntropyModel for VQ indices (rate estimation & coding).

    NOTE on Coding:
    - Training rate uses context-dependent logits for accuracy.
    - Compress/Decompress uses context-free logits (derived only from z_hat)
      for compatibility with non-sequential tfc.CategoricalEntropyModel coding,
      making the actual coded bitrate suboptimal compared to training estimate.
    """
    def __init__(self,
                 num_embeddings, # Number of VQ codes = num categories
                 latent_dim, # Channels of main latent y/quantized_y
                 hyper_latent_dim=32,
                 hyper_filters=96,
                 context_filters=96,
                 joint_filters=128,
                 name="hyperprior_sc_tfc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim # Needed for hyper-encoder input shape
        self.hyper_latent_dim = hyper_latent_dim
        self.hyper_filters = hyper_filters
        self.context_filters = context_filters
        self.joint_filters = joint_filters

        # Sub-models (defined in build)
        self.hyper_encoder_model = None
        self.hyper_decoder_model = None
        self.context_model = None
        self.joint_parameter_predictor = None
        # === New: Predictor using only z (for context-free coding) ===
        self.context_free_predictor = None

        # === TFC Layers ===
        # For hyper-latent z (continuous -> quantized -> coded)
        self.entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
        # For VQ indices (discrete symbols -> coded using predicted probabilities)
        self.categorical_entropy_model = tfc.CategoricalEntropyModel(
            prior_shape=(num_embeddings,), coding_rank=3, # B, H, W indices
            bottleneck_dtype=None, use_likelihood_bound=False
        )
        # === End TFC Layers ===

        self._built = False

    def build(self, main_latent_shape):
        if self._built: return
        main_latent_shape = tf.TensorShape(main_latent_shape) # Should be (Hl, Wl, Cl)
        print(f"Building HyperpriorSC with main_latent_shape: {main_latent_shape}")
        policy = tf.keras.mixed_precision.global_policy()
        compute_dtype = policy.compute_dtype

        # --- 1. Hyper-Encoder (y -> z) ---
        hyper_enc_input = tf.keras.Input(shape=main_latent_shape, name="hyper_enc_input_y")
        x = tf.cast(hyper_enc_input, compute_dtype)
        # Optional: Add non-linearity like tf.abs(x) here
        x = tf.keras.layers.Conv2D(self.hyper_filters, 3, 1,'same', activation='relu', name='hyper_enc_conv1')(x)
        x = tf.keras.layers.Conv2D(self.hyper_filters, 5, 2,'same', activation='relu', name='hyper_enc_conv2')(x)
        x = tf.keras.layers.Conv2D(self.hyper_filters, 5, 2,'same', activation='relu', name='hyper_enc_conv3')(x)
        z = tf.keras.layers.Conv2D(self.hyper_latent_dim, 3, 1, 'same', name='hyper_enc_output_z')(x)
        self.hyper_encoder_model = tf.keras.Model(hyper_enc_input, z, name="hyper_encoder")
        print(f" HyperpriorSC Encoder built. Output 'z' shape: {z.shape}")
        hyper_latent_z_shape = z.shape[1:]

        # --- 2. Hyper-Decoder (z_hat -> params_from_z) ---
        hyper_dec_input = tf.keras.Input(shape=hyper_latent_z_shape, name="hyper_dec_input_z")
        x = tf.cast(hyper_dec_input, compute_dtype)
        x = tf.keras.layers.Conv2DTranspose(self.hyper_filters, 5, 2, 'same', activation='relu', name='hyper_dec_deconv1')(x)
        x = tf.keras.layers.Conv2DTranspose(self.hyper_filters, 5, 2, 'same', activation='relu', name='hyper_dec_deconv2')(x)
        params_from_z = tf.keras.layers.Conv2D(self.hyper_filters, 3, 1, 'same', name='hyper_dec_params_from_z')(x)
        self.hyper_decoder_model = tf.keras.Model(hyper_dec_input, params_from_z, name="hyper_decoder")
        print(f" HyperpriorSC Decoder built. Output 'params_from_z' shape: {params_from_z.shape}")
        params_from_z_shape = params_from_z.shape[1:]

        # --- 3. Context Model (quantized_y -> context_features) ---
        ctx_input = tf.keras.Input(shape=main_latent_shape, name="context_input_quantized_y")
        x = tf.cast(ctx_input, compute_dtype)
        x = MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', activation='relu', name='ctx_masked_conv1')(x)
        x = MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', activation='relu', name='ctx_masked_conv2')(x)
        context_features = MaskedConv2D(mask_type='A', filters=self.context_filters, kernel_size=5, padding='same', name='ctx_features')(x)
        self.context_model = tf.keras.Model(ctx_input, context_features, name="context_model")
        print(f" HyperpriorSC Context Model built. Output 'context_features' shape: {context_features.shape}")
        context_features_shape = context_features.shape[1:]

        # --- 4. Joint Parameter Predictor (params_from_z + context_features -> logits) ---
        joint_input_z = tf.keras.Input(shape=params_from_z_shape, name="joint_input_params_z")
        joint_input_ctx = tf.keras.Input(shape=context_features_shape, name="joint_input_ctx")
        input_z_casted = tf.cast(joint_input_z, compute_dtype)
        input_ctx_casted = tf.cast(joint_input_ctx, compute_dtype)
        joint_features = tf.keras.layers.Concatenate(axis=-1)([input_z_casted, input_ctx_casted])
        x = tf.keras.layers.Conv2D(self.joint_filters, 1, 1,'same', activation='relu', name='joint_conv1')(joint_features)
        x = tf.keras.layers.Conv2D(self.joint_filters, 3, 1,'same', activation='relu', name='joint_conv2')(x)
        logits_context = tf.keras.layers.Conv2D(self.num_embeddings, 1, 1, 'same', activation=None,
                                                dtype='float32', name='joint_output_logits_context')(x)
        self.joint_parameter_predictor = tf.keras.Model(
            inputs=[joint_input_z, joint_input_ctx], outputs=logits_context, name="joint_parameter_predictor")
        print(f" HyperpriorSC Joint Predictor built. Output 'logits_context' shape: {logits_context.shape}")

        # --- 5. Context-Free Predictor (params_from_z -> logits) ---
        # Simple 1x1 conv predictor taking only hyperprior features
        cf_input_z = tf.keras.Input(shape=params_from_z_shape, name="cf_input_params_z")
        cf_input_z_casted = tf.cast(cf_input_z, compute_dtype)
        # Reuse joint_filters size? Or use a smaller dedicated size? Let's reuse for simplicity.
        cf_x = tf.keras.layers.Conv2D(self.joint_filters, 1, 1, 'same', activation='relu', name='cf_conv1')(cf_input_z_casted)
        logits_context_free = tf.keras.layers.Conv2D(self.num_embeddings, 1, 1, 'same', activation=None,
                                                    dtype='float32', name='cf_output_logits')(cf_x)
        self.context_free_predictor = tf.keras.Model(
             inputs=cf_input_z, outputs=logits_context_free, name="context_free_predictor")
        print(f" HyperpriorSC Context-Free Predictor built. Output 'logits_context_free' shape: {logits_context_free.shape}")


        self._built = True

    @tf.function
    def get_logits_with_context(self, y, quantized_y, training=False):
        """Gets logits using both hyperprior and context model (for training rate)."""
        if not self._built: self.build(tf.shape(y)[1:])
        y_input_he = tf.stop_gradient(y) if training else y
        z = self.hyper_encoder_model(y_input_he, training=training)
        # Need z_hat for hyper_decoder. Use EB in inference mode to get it non-destructively.
        z_hat, _ = self.entropy_bottleneck(tf.cast(z, tf.float32), training=False)
        z_hat_compute = tf.cast(z_hat, y.dtype)

        params_from_z = self.hyper_decoder_model(z_hat_compute, training=training)
        context_features = self.context_model(quantized_y, training=training)

        policy = tf.keras.mixed_precision.global_policy()
        params_from_z_casted = tf.cast(params_from_z, policy.compute_dtype)
        context_features_casted = tf.cast(context_features, policy.compute_dtype)

        final_logits = self.joint_parameter_predictor([params_from_z_casted, context_features_casted], training=training)
        return final_logits # float32

    @tf.function
    def get_logits_context_free(self, z_hat):
        """Gets logits using only z_hat (for context-free coding)."""
        if not self._built: raise RuntimeError("Model must be built before calling get_logits_context_free")
        # Ensure z_hat has the compute dtype expected by hyper_decoder
        z_hat_compute = tf.cast(z_hat, tf.keras.mixed_precision.global_policy().compute_dtype)
        params_from_z = self.hyper_decoder_model(z_hat_compute, training=False)
        # Predict logits using only params_from_z
        logits = self.context_free_predictor(params_from_z, training=False)
        return tf.cast(logits, tf.float32) # Ensure float32


    @tf.function
    def call(self, inputs, training):
        """
        Forward pass for training. Computes rate using context-dependent logits.

        Args:
            inputs (tuple): (y, quantized_y, indices). y and quantized_y have compute_dtype.
                            indices have int dtype.
            training (bool): Training mode.

        Returns:
            tf.Tensor: Rate loss (bits per element) calculated using context.
        """
        if not isinstance(inputs, tuple) or len(inputs) != 3:
            raise ValueError("HyperpriorSC call expects (y, quantized_y, indices)")
        y, quantized_y, indices = inputs

        # Get context-dependent logits for accurate rate calculation
        logits = self.get_logits_with_context(y, quantized_y, training=training)

        # Calculate rate using TFC CategoricalEntropyModel
        indices_int = tf.cast(indices, tf.int32)
        logits_f32 = tf.cast(logits, tf.float32)
        _, bits = self.categorical_entropy_model(indices_int, logits_f32, training=training)

        # Average bits over all elements
        num_elements = tf.cast(tf.reduce_prod(tf.shape(indices_int)), tf.float32)
        if num_elements == 0: return tf.constant(0.0, dtype=tf.float32)
        rate_loss_bits_per_element = tf.reduce_sum(bits) / num_elements

        return rate_loss_bits_per_element

    # === COMPRESSION / DECOMPRESSION using TFC layers ===

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32), # y (float32)
        tf.TensorSpec(shape=(None, None, None), dtype=tf.int64) # indices (int64)
    ])
    def compress(self, y, indices):
        """
        Compresses y (via z) and indices using TFC layers.
        Uses context-free probabilities for indices.
        """
        print("Tracing HyperpriorSC compress...")
        # Ensure y is compute dtype for hyper-encoder
        y_compute = tf.cast(y, tf.keras.mixed_precision.global_policy().compute_dtype)

        # --- Compress z ---
        z = self.hyper_encoder_model(y_compute)
        z_shape = tf.shape(z)
        # EB compress expects float32
        z_strings = self.entropy_bottleneck.compress(tf.cast(z, tf.float32))

        # --- Compress indices using context-free probs ---
        # Need z_hat to get context-free probs
        z_spatial_shape = z_shape[1:-1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_spatial_shape)
        z_hat = tf.reshape(z_hat, z_shape) # float32

        # Get context-free logits using z_hat
        logits_cf = self.get_logits_context_free(z_hat) # float32

        # Compress indices using categorical model and context-free logits
        indices_int = tf.cast(indices, tf.int32)
        indices_strings = self.categorical_entropy_model.compress(indices_int, logits_cf)

        # Shape of y is needed for reshaping during potential context model usage in decoder
        y_shape = tf.shape(y)

        return [z_strings, indices_strings], [z_shape, y_shape] # Return y shape for consistency

    @tf.function(input_signature=[
        [tf.TensorSpec(shape=(None,), dtype=tf.string), # z_strings
         tf.TensorSpec(shape=(None,), dtype=tf.string)],# indices_strings
        [tf.TensorSpec(shape=(4,), dtype=tf.int32),     # z_shape
         tf.TensorSpec(shape=(4,), dtype=tf.int32)]     # y_shape (not directly used here)
        ])
    def decompress(self, strings, shapes):
        """
        Decompresses z_strings and indices_strings using TFC layers.
        Uses context-free probabilities for indices.
        """
        print("Tracing HyperpriorSC decompress...")
        z_strings, indices_strings = strings
        z_shape, _ = shapes # y_shape isn't needed here, only z_shape for EB

        # --- Decompress z ---
        z_spatial_shape = z_shape[1:-1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_spatial_shape)
        z_hat = tf.reshape(z_hat, z_shape) # float32 output

        # --- Decompress indices ---
        # Get context-free logits using z_hat
        logits_cf = self.get_logits_context_free(z_hat) # float32

        # Decompress indices using categorical model and context-free logits
        indices_decoded = self.categorical_entropy_model.decompress(indices_strings, logits_cf)
        indices_decoded = tf.cast(indices_decoded, tf.int64) # Cast back to int64 like original

        # Return only the indices, as z_hat isn't needed by the main AE decoder
        # The caller (FlowVQAutoencoder) will use indices to reconstruct y_hat
        return indices_decoded