import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
from pathlib import Path
import PIL.Image
import tensorflow as tf
from tensorflow.keras.applications import VGG19
import time

# Set image paths
content_path = Path("./input/input_content.jpg")
style_path = Path("./input/input_style.jpg")
output_prefix = Path("./output")


def load_image(image_path):
    """
    Load image as a Tensor from file, scaling the pixels to a range of [0,1) and
    resizing the image to a maximum of 512 pixels (width). Supports images in
    jpeg, png, gif and bmp format.
    -----------
    Args:
        image_path ([str]): path to image to convert into array.
    -----------
    Returns:
        [tensorflow.Tensor]: Tensor in RGB encoding of the image.
    """

    max_dim = 512
    img = tf.io.read_file(str(image_path.resolve()))
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def imshow(image, title=None):
    """
    Show image from Tensor.
    -----------
    Args:
        image ([tensorflow.Tensor]): image converted from Tensor.
        title ([str], optional): image title. Defaults to None.
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    plt.axis('off')

    if title:
        plt.title(title)


def tensor_to_image(tensor):
    """
    Converts a tensorflow.Tensor to image.
    -----------
    Args:
        tensor (tensorflow.Tensor): tensor of image
    -----------
    Returns:
        [PIL.Image]: image object created from tensor.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def vgg_layers(layer_names):
    """
    Creates model from VGG19 to return a list of intermediate output
    values based on pretrained kernels based on imagenet data.
    -----------
    Args:
        layer_names (list): List of layer names.
    -----------
    Returns:
        [tensorflow.Keras.Model]: Model containing a list of intermediate representations
        
    """
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model


def gram_matrix(input_tensor):
    """
    Generate Gram Matrix from input tensor.
    -----------
    Args:
        input_tensor ([tensorflow.Tensor]): Tensor of intermediate representations of content layers.
    -----------
    Returns:
        [tensorflow.Tensor]: Gram matrix of content layers.
    """

    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / (num_locations)


def clip_0_1(image):
    """
    Keeps pixels in image between 0 and 1.
    -----------
    Args:
        image ([tensorflow.Tensor]): image represented as tensor.
    -----------
    Returns:
        [tensorflow.Tensor] 
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """
        Creates model for Neural Style Transfer based on VGG19.
        -----------
        Args:
            inputs ([tensorflow.Tensor]): image loaded as tensor.
        -----------
        Returns:
            [dict]: dictionary containing the gram matrix for style representation
            and tensor containing style representation.
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [
            gram_matrix(style_output) for style_output in style_outputs
        ]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers,
                                           content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }

        return {'content': content_dict, 'style': style_dict}


def style_content_loss(outputs):
    """
    Calculates total loss for content and style by performing element-wise summation
    of each intermediate representation of content and style, and then multiplying by 
    the selected weight for each.
    -----------
    Args:
        outputs ([dict]): tensors with intermediate representations of style and content.
    -----------
    Returns:
        [tensorflow.Tensor]: tensor containing total loss value.
    """

    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([
        tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
        for name in style_outputs.keys()
    ])
    style_loss *= style_weight * num_style_layers

    content_loss = tf.add_n([
        tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
        for name in content_outputs.keys()
    ])
    content_loss *= content_weight * num_content_layers

    loss = style_loss + content_loss

    return loss


@tf.function()
def train_step(image):
    """
    Performs a training iteration of Neural Style Transfer.
    -----------
    Args:
        image ([tensorflow.Tensor]): white noise image.
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


# Load images
content_image = load_image(content_path)
style_image = load_image(style_path)

# Visualize images
#plt.subplot(1, 2, 1)
#imshow(content_image, 'Content Image')

#plt.subplot(1, 2, 2)
#imshow(style_image, 'Style Image')

# set weights for losses
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

# keep intermediate layers to represent content and style
content_layers = ['block5_conv2']

style_layers = [
    'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
    'block5_conv1'
]

# set number of layers for content and style
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# instantiate Style Transfer Model
extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# create tensorflow.Variable as white noise to contain new image
image = tf.Variable(content_image)
# set the optimizer - lbfgs recommended by paper
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# perform training operation
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='', flush=True)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))

# save results
file_name = str(output_prefix.resolve()) + '/stylized-image.png'
tensor_to_image(image).save(file_name)