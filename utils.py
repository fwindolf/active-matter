import keras
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def to_classes(data, n_classes=4, ordering='channel_last', threshold=0.00):
    """
    Convert data with per-pixel class labels to a representation where the labels are shown
    in the channels dimensions.
    Args:
        data: Input image-based data that should be converted
        n_classes: classes that are considered for the output data. Corresponds to the number of channels of output
        ordering: The data ordering of the image data [channel_last, channel_first]
        threshold: Only consider pixels above the threshold [0...1] for the output
    Returns:
        A vector inflated to n_classes in the channels dimension that contains 1 everywhere where the class that the
        channel represents is active.
    """
    output = data
    output[output <= threshold] = 0
    if ordering is 'channel_first':
        assert(data.shape[-3] == 1)
        output = np.argmax(data, axis=-3)
        output = keras.utils.to_categorical(output, num_classes=n_classes)
        output = np.moveaxis(output, -1, -3)
    else:
        assert(data.shape[-1] == 1)
        output = np.argmax(data, axis=-1)
        output = keras.utils.to_categorical(output, num_classes=n_classes)
    return output


def estimate_batchsize(model, memory, timesteps=1):
    """
    Estimate the maximum batch size usable for the model
    https://stackoverflow.com/questions/46654424/how-to-calculate-optimal-batch-size

    Args:
        model: The model that should be trained
        data: Input data without the batch dimension
        memory: Available memory in MB
    Returns: 
        An estimate for the batchsize that should work with the model
    """
    # Calculate the GPU memory in bytes
    memory_bytes = 1000000 * memory
    
    # Get the number of trainable parameters 
    # https://stackoverflow.com/questions/45046525/keras-number-of-trainable-parameters-in-model
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    model_bytes = trainable_count * np.dtype(np.float32).itemsize
    
    # Get the size of every tensor in the model
    tensor_bytes = 0
    for l in model.layers:
        shape = l.input_shape
        if "concatenate" in l.name:
            continue
            
        # tensor in this layer is the bytes in input_shape
        data_bytes = np.prod(shape[-3:]) * np.dtype(np.float32).itemsize        
        if "lst" in l.name:
            data_bytes *= timesteps

        tensor_bytes += data_bytes

    batch_bytes = (memory_bytes - model_bytes) / tensor_bytes
    return batch_bytes / np.dtype(np.float32).itemsize


def show_image(img, greyscale=False, ordering='channel_last'):
    """
    Show/Plot an image from various formats
    """
    # to channel_last if channel exists
    if len(img.shape) > 2 and ordering == 'channel_first':
        img = np.moveaxis(img, 0, -1)

    if img.dtype == np.uint8 and np.max(img) <= 1:
        img = img * 255

    img = np.squeeze(img)
    if greyscale or len(img.shape) < 3:
        plt.imshow(img, cmap='gray')
        return
    
    plt.imshow(img)

def show_label(lbl, ignore_background=True, ordering='channel_last'):
    """
    Show/Plot a label from various formats with various numbers of channels
    """
    assert(len(lbl.shape) > 2) # make sure channel exists
    
    # to channel_last
    if ordering == 'channel_first':
        lbl = np.moveaxis(lbl, 0, -1)
    
    # check if channels exist
    if ignore_background and lbl.shape[-1] > 1:
        lbl = lbl[..., 1:]
    
    if lbl.dtype == np.uint8:
        lbl = (lbl * 255)

    lbl = np.squeeze(lbl)
    # greyscale
    if len(lbl.shape) < 2 or lbl.shape[-1] == 1:
        plt.imshow(lbl, cmap='gray')
        return
    
    # 3 colors
    if lbl.shape[-1] == 3:        
        plt.imshow(lbl)
        return 
    
    # more colors
    colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 0, 1)]
    colors = colors[int(ignore_background):] # leave out black if ignore_background is True
    
    # Add 
    output = np.zeros(lbl.shape[:-1], dtype=np.float)
    for c in range(lbl.shape[-1]):
        output += lbl[..., c] * colors[c]
    raise NotImplementedError("No extra colors available")


def show_sequence(data, ordering='channel_last'):
    """
    Visualizes one batch of sequence data.
    Args:
        data: Data batch containing a tuple of X and Y data with dimensions [b, stacked_size, c, h, w]
        stacked_size: Number of images stacked ontop of each other to create the data.
        ordering: Ordering of the data, channel_first means [c, h, w], channel_last means [h, w, c]
    Return:
        A figure containing the plots with horizontally stacked images.
    """
    xb, yb = data
    batch_size = xb.shape[0]
    stacked_size = xb.shape[1]
        
    fig = plt.figure(figsize=(5 * stacked_size, 5 * 2 * batch_size))
    for i in range(batch_size):
        x = xb[i]
        for j in range(stacked_size):
            fig.add_subplot(2 * batch_size, stacked_size, stacked_size * (2 * i) + j + 1) 
            show_image(x[j])
        
        if yb[i] is not None:
            y = yb[i]
        else:
            y = np.zeros_like(xb[i])
        
        for j in range(stacked_size):
            fig.add_subplot(2 * batch_size, stacked_size, stacked_size * (2 * i + 1) + j + 1)
            if ordering == 'channel_first':
                y[j] = np.moveaxis(y[j], 0, -1)
            
            if y.shape[-1] == 1:
                show_image(y[j])
            else:
                show_label(y[j])

    return fig
    
def show_stacked(data, stacked_size=3, ordering='channel_last'):
    """
    Visualizes one batch of stacked data.
    Args:
        data: Data batch containing a tuple of X [b, stacked_size, 1, h, w] and Y [b, 1, n_classes, h, w]
        stacked_size: Number of images stacked ontop of each other to create the data.
        ordering: Ordering of the data, channel_first means [c, h, w], channel_last means [h, w, c]
    Return:
        A figure containing the plots with horizontally stacked X images with the Y image at the right
    """
    xb, yb = data
    batch_size = len(xb)
    
    fig = plt.figure(figsize=(5* (stacked_size + 1), 5*batch_size))
    for i in range(batch_size):
        x = xb[i]
        for j in range(stacked_size):
            plt.subplot(batch_size, (stacked_size + 1), (stacked_size + 1) * i + j + 1) 
            if ordering == 'channel_first':
                show_image(x[j])
            else:
                show_image(x[..., j])
            
        fig.add_subplot(batch_size, (stacked_size + 1), (stacked_size + 1) * i + stacked_size + 1)
        if yb[i] is None:
            y = np.zeros_like(xb[i])
        else:
            y = yb[i]            

        
        show_label(y)
    
    return fig

def show_pair(data, labeled=True, ordering='channel_last'):
    """
    Visualizes one batch of paired data.
    Args:
        data: Tuple of x, y data.
        labeled: Flag if the data is labeled
        ordering: Ordering of the data, channel_first means [c, h, w], channel_last means [h, w, c]
    Returns:
        A figure containing the plots with vertically stacked batches of X, Y images.
    """
    xb, yb = data
    batch_size = len(xb)
    
    fig = plt.figure(figsize=(5 * 2, 5 * batch_size))
    for i in range(batch_size):
        fig.add_subplot(batch_size, 2, (2 * i) + 1) 
        show_image(xb[i], ordering=ordering)
            
        fig.add_subplot(batch_size, 2, (2 * i) + 2) 
        if yb[i] is None:
            y = np.zeros_like(xb[i])
        else:
            y = yb[i]            

        if labeled:
            show_label(y, ordering=ordering)
        else:
            show_image(y, ordering=ordering)
    
    return fig

def compare_pair(data, labeled=True, ordering='channel_last'):
    """
    Visualizes one batch of paired data.
    Args:
        data: Tuple of x, y_pred, y data.
        labeled: Flag if the data is labeled
        ordering: Ordering of the data, channel_first means [c, h, w], channel_last means [h, w, c]
    Returns:
        A figure containing the plots with vertically stacked batches of X, Y_PRED, Y images.
    """
    xb, yb_pred, yb = data
    batch_size = len(xb)
    
    fig = plt.figure(figsize=(5 * 3, 5 * batch_size))
    for i in range(batch_size):
        fig.add_subplot(batch_size, 3, (3 * i) + 1) 
        show_image(xb[i], ordering=ordering)

        fig.add_subplot(batch_size, 3, (3 * i) + 2) 
        if labeled:
            show_label(yb_pred[i], ordering=ordering)
        else:
            show_image(yb_pred[i], ordering=ordering)
            
        fig.add_subplot(batch_size, 3, (3 * i) + 3) 
        if yb[i] is None:
            y = np.zeros_like(xb[i])
        else:
            y = yb[i]            
        
        if labeled:
            show_label(y, ordering=ordering)
        else:
            show_image(y, ordering=ordering)
        
    return fig


def compare_sequence(data, ordering='channel_last'):
    """
    Visualizes one batch of sequence data.
    Args:
        data: Data batch containing a tuple of X and Y data with dimensions [b, stacked_size, c, h, w]
        stacked_size: Number of images stacked ontop of each other to create the data.
        ordering: Ordering of the data, channel_first means [c, h, w], channel_last means [h, w, c]
    Return:
        A figure containing the plots with horizontally stacked images.
    """
    xb, ypb, yb = data
    batch_size = xb.shape[0]
    stacked_size = xb.shape[1]
        
    fig = plt.figure(figsize=(5 * stacked_size, 5 * 3 * batch_size))
    for i in range(batch_size):
        # Input
        x = xb[i]
        for j in range(stacked_size):
            fig.add_subplot(3 * batch_size, stacked_size, stacked_size * (3 * i) + j + 1) 
            show_image(x[j])
    
        # Prediction
        yp = ypb[i]
        for j in range(stacked_size):
            fig.add_subplot(3 * batch_size, stacked_size, stacked_size * (3 * i + 1) + j + 1)
            if ordering == 'channel_first':
                yp[j] = np.moveaxis(yp[j], 0, -1)
            
            if yp.shape[-1] == 1:
                show_image(yp[j])
            else:
                show_label(yp[j])

        # Label
        if yb[i] is not None:
            y = yb[i]
        else:
            y = np.zeros_like(xb[i])

        for j in range(stacked_size):
            fig.add_subplot(3 * batch_size, stacked_size, stacked_size * (3 * i + 2) + j + 1)
            if ordering == 'channel_first':
                y[j] = np.moveaxis(y[j], 0, -1)
            
            if y.shape[-1] == 1:
                show_image(y[j])
            else:
                show_label(y[j])

    return fig