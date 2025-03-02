import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import pandas



class Length(layers.Layer):
    # compute thr norme if the input vector
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    
    # return the output shape
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    # retourne la configuration de la couche 
    def get_config(self):
        config = super(Length, self).get_config()
        return config

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + K.epsilon())

#------------------------------------------Class Capsule layer ----------------------------------------------------------------------------------
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings = 3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsule      # Output capsules 2
        self.dim_capsule = dim_capsule       # Dimension of each output capsule 16
        self.routings = routings       # Number of routing iterations 3
        self.input_num_capsule = None        # To be defined by input shape 36448
        self.input_dim_capsule = None        # To be defined by input shape 8

    def build(self, input_shape):
        # Define input shape properties
        self.input_num_capsule = input_shape[1]  # Number of input capsules (36448)
        self.input_dim_capsule = input_shape[2]  # Dimension of input capsules (8)

        # Initialize weights for transformation
        self.W = self.add_weight(
            shape=(1, self.input_num_capsule, self.num_capsules, self.dim_capsule, self.input_dim_capsule),
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        ) # (1, n, 2, 16, 8)
        super(CapsuleLayer, self).build(input_shape)
        
    def call(self, inputs): # (None, n, 8)
        # Expand and tile the inputs
        u_hat = tf.expand_dims(inputs, -1) # (None, n, 8, 1)
        u_hat = tf.expand_dims(u_hat, 2) # (None, n, 1, 8, 1)
        u_hat = tf.tile(u_hat, [1, 1, self.num_capsules, 1, 1]) # (None, n, 2, 8, 1)
        u_hat = tf.matmul(self.W, u_hat) # (1, n, 2, 16, 8) * (None, n, 2, 8, 1) = (None, n, 2, 16, 1)

        # Initialize log prior probabilities b to zero
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.input_num_capsule, self.num_capsules, 1, 1], dtype=np.float32)
        # (None, n, 2, 1, 1)
        for i in range(self.routings):
            # Compute coupling coefficients using softmax
            c = tf.nn.softmax(b, axis=2) # (None, n, 2, 1, 1)
            
            # Compute weighted sum of the prediction vectors
            outputs = tf.multiply(c, u_hat) # (None, n, 2, 1, 1)*(None, n, 2, 16, 1) =(None, n, 2, 16, 1) # multiplication elmt par elemt

            s = tf.reduce_sum(outputs, axis=1, keepdims=True) # (None, 1, 2, 16, 1)

            # Squash the outputs
            v = squash(s) # (None, 1, 2, 16, 1)

            if i < self.routings - 1:
                # Update log prior probabilities (routing weights) for the next iteration
                v_tiled = tf.tile(v, [1, self.input_num_capsule, 1, 1, 1]) # (None, n, 2, 16, 1)
                v_matmul = tf.matmul(u_hat, v_tiled, transpose_a=True) # (1, 16, 2, n, None) * (None, n, 2, 16, 1) = (None, n, 2, 1, 1)  
                b = tf.add(b, v_matmul) # (None, n, 2, 1, 1)
                
        return tf.reshape(v, shape = (-1, 2, 16)) # (None, 1, 2, 16, 1)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsules, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsules,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#-------------------------PrimaryCpaLayer-------------------------------------------------------

def PrimaryCap3D(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    outputs = layers.Conv3D(filters=dim_capsule * n_channels, 
                           kernel_size=kernel_size, 
                           strides=strides, 
                           padding=padding, 
                           name= 'primaryCap')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(outputs) # (none, n, 8)
    outputs = layers.Lambda(squash, name='primarycap_squash')(outputs)
    return outputs


#----------------------------------------build Architecture of the Model----------------------
K.set_image_data_format('channels_last')
def CapsNet3D(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    
    # VGG-like Block 1
    conv1 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_1')(x)
    conv1 = layers.BatchNormalization(name='batch_norm1_1')(conv1)
    
    conv2 = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='conv1_2')(conv1)
    conv2 = layers.BatchNormalization(name='batch_norm1_2')(conv2)
    
    pooled1 = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='max_pool1')(conv2)
    
    # VGG-like Block 2
    conv3 = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_1')(pooled1)
    conv3 = layers.BatchNormalization(name='batch_norm2_1')(conv3)
    
    conv4 = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_2')(conv3)
    conv4 = layers.BatchNormalization(name='batch_norm2_2')(conv4)
    
    pooled2 = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='max_pool2')(conv4)

    # VGG-like Block 3
    conv5 = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_1')(pooled2)
    conv5 = layers.BatchNormalization(name='batch_norm3_1')(conv5)
    
    conv6 = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_2')(conv5)
    conv6 = layers.BatchNormalization(name='batch_norm3_2')(conv6)
    
    conv7 = layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3_3')(conv6)
    conv7 = layers.BatchNormalization(name='batch_norm3_3')(conv7)
    
    pooled3 = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='max_pool3')(conv7)

    # VGG-like Block 4
    conv8 = layers.Conv3D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_1')(pooled3)
    conv8 = layers.BatchNormalization(name='batch_norm4_1')(conv8)
    
    conv9 = layers.Conv3D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_2')(conv8)
    conv9 = layers.BatchNormalization(name='batch_norm4_2')(conv9)
    
    conv10 = layers.Conv3D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4_3')(conv9)
    conv10 = layers.BatchNormalization(name='batch_norm4_3')(conv10)
    
    #pooled4 = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='max_pool4')(conv10)
    
    # 1st Capsule Layer
    primarycaps = PrimaryCap3D(conv10, dim_capsule=16, n_channels=8, kernel_size=2, strides=1, padding='valid')
    
    # Capsule Layer
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
    
    # Length Layer to compute the norm of capsule outputs
    out_caps = Length(name='capsnet')(digitcaps)
    
    y = layers.Input(shape=(n_class,))
    
    # Models for training and evaluation
    train_model = models.Model([x, y], [out_caps])
    eval_model = models.Model(x, [out_caps])
    
    return train_model, eval_model

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


#-----------------------------plot from csv File---------------------------------------------------
def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


#--------------------------------------------Convert Data from Mesh to Voxels and augement it----------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.ndimage import zoom, rotate, shift
import trimesh
import numpy as np
import random

def load_obj_as_voxel(filepath, grid_size):
    # Load the mesh from the .obj file
    mesh = trimesh.load(filepath)

    # Convert the mesh to a voxel grid
    voxel_grid = mesh.voxelized(pitch=mesh.extents.max()/grid_size)

    # Convert voxel grid to a numpy array
    voxel_data = voxel_grid.matrix.astype(np.float32)

    # Resize voxel data to grid_size x grid_size x grid_size if necessary
    current_shape = voxel_data.shape
    if current_shape != (grid_size, grid_size, grid_size):
        zoom_factors = [g/c for g, c in zip((grid_size, grid_size, grid_size), current_shape)]
        voxel_data = zoom(voxel_data, zoom_factors, order=0)  # Nearest-neighbor interpolation

    return voxel_data

def augment_voxel_data(voxel):
    augmented_data = []

    # Original voxel
    augmented_data.append(voxel)

    # Rotation on each axis (90, 180, 270 degrees)
    for angle in [90, 180, 270]:
        rotated_x = rotate(voxel, angle, axes=(1, 2), reshape=False, order=0)
        rotated_y = rotate(voxel, angle, axes=(0, 2), reshape=False, order=0)
        rotated_z = rotate(voxel, angle, axes=(0, 1), reshape=False, order=0)
        augmented_data.extend([rotated_x, rotated_y, rotated_z])

    # Shifting the voxel grid along different axes
    for shift_val in [-2, 2]:
        shifted_x = shift(voxel, shift=[shift_val, 0, 0], order=0)
        shifted_y = shift(voxel, shift=[0, shift_val, 0], order=0)
        shifted_z = shift(voxel, shift=[0, 0, shift_val], order=0)
        augmented_data.extend([shifted_x, shifted_y, shifted_z])

    # Adding random binary noise
    noise = np.random.binomial(1, 0.02, voxel.shape)
    noisy_voxel = voxel + noise
    noisy_voxel = np.clip(noisy_voxel, 0, 1)
    augmented_data.append(noisy_voxel)

    # Adding Gaussian noise
    gaussian_noise = np.random.normal(0, 0.02, voxel.shape)
    noisy_voxel_gaussian = voxel + gaussian_noise
    noisy_voxel_gaussian = np.clip(noisy_voxel_gaussian, 0, 1)
    augmented_data.append(noisy_voxel_gaussian)

    # Random flipping along different axes
    flipped_x = np.flip(voxel, axis=0)
    flipped_y = np.flip(voxel, axis=1)
    flipped_z = np.flip(voxel, axis=2)
    augmented_data.extend([flipped_x, flipped_y, flipped_z])

    # Elastic deformation (simple scaling for example)
    scale_factors = [0.8, 1.2]
    for scale in scale_factors:
        scaled_voxel = zoom(voxel, scale, order=1)
        scaled_voxel = zoom(scaled_voxel, 1/scale, order=1)  # Rescale back to original size
        augmented_data.append(scaled_voxel)

    # Random cropping and resizing to the original size
    crop_size = random.randint(voxel.shape[0] // 2, voxel.shape[0])
    start = random.randint(0, voxel.shape[0] - crop_size)
    cropped_voxel = voxel[start:start+crop_size, start:start+crop_size, start:start+crop_size]
    cropped_voxel_resized = zoom(cropped_voxel, (voxel.shape[0] / crop_size,) * 3, order=0)
    augmented_data.append(cropped_voxel_resized)

    # Random rotation with small angles
    for angle in np.random.uniform(-15, 15, size=3):
        rotated_small_angle = rotate(voxel, angle, axes=(1, 2), reshape=False, order=1)
        augmented_data.append(rotated_small_angle)

    return augmented_data

def load_voxel_data(data_dir, grid_size=30, test_size=0.3, test=False):
    voxel_data = []
    labels = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.obj'):
                    file_path = os.path.join(class_dir, file_name)
                    voxel = load_obj_as_voxel(file_path, grid_size=grid_size)
                    print(file_name, class_name)
                    if test==True:
                        voxel_data.append(voxel)
                        labels.append(class_name)
                    elif voxel is not None:
                        augmented_voxels = augment_voxel_data(voxel)
                        voxel_data.extend(augmented_voxels)
                        labels.extend([class_name] * len(augmented_voxels))

    # Convert lists to numpy arrays
    voxel_data = np.array(voxel_data)
    labels = np.array(labels)

    print(f"Voxel data shape after augmentation: {voxel_data.shape}")
    print(f"Labels shape after augmentation: {labels.shape}")

    voxel_data = voxel_data.reshape(-1, grid_size, grid_size, grid_size, 1)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.eye(len(label_encoder.classes_))[labels]  # One-hot encode
    
    # To get the mapping of classes to their codes
    class_mapping = {class_label: code for code, class_label in enumerate(label_encoder.classes_)}

    # Print the mapping
    for class_label, code in class_mapping.items():
        print(f"Class '{class_label}' is encoded as {code}")

    if test:
        return voxel_data, labels
    else:
        x_train, x_val, y_train, y_val = train_test_split(voxel_data, labels, test_size=test_size, random_state=42)
        return (x_train, y_train), (x_val, y_val)




