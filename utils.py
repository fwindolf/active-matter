import keras
import numpy as np
import matplotlib.pyplot as plt

def to_classes(prediction, n_classes=4, ordering='channel_first', threshold=0.00):
    output = prediction
    output[output <= threshold] = 0
    if ordering is 'channel_first':
        output = np.argmax(prediction, axis=-3)
        output = keras.utils.to_categorical(output, num_classes=n_classes)
        output = np.moveaxis(output, -1, -3)
    else:
        output = np.argmax(prediction, axis=-1)
        output = keras.utils.to_categorical(output, num_classes=n_classes)
    return output

def show_image(img, greyscale=False, ordering='channel_first'):
    """
    Show/Plot an image from various formats
    """
    # to channel_last if channel exists
    if len(img.shape) > 2 and ordering == 'channel_first':
        img = np.moveaxis(img, 0, -1)

    if img.dtype == np.uint8:
        img = img * 255

    img = np.squeeze(img)
    if greyscale or len(img.shape) < 3:
        plt.imshow(img, cmap='gray')
        return
    
    plt.imshow(img)

def show_label(lbl, ignore_background=True, ordering='channel_first'):
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


def show_sequence(data, stacked_size=3, ordering='channel_first'):
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
    batch_size = len(xb)
    
    fig = plt.figure(figsize=(5 * stacked_size, 5 * 2 * batch_size))
    for i in range(batch_size):
        x = xb[i]
        for j in range(stacked_size):
            fig.add_subplot(2 * batch_size, stacked_size, stacked_size * (2 * i) + j + 1) 
            if ordering == 'channel_first':
                show_image(x[j])
            else:
                show_image(x[..., j])
        
        if yb[i] is not None:
            y = yb[i]
        else:
            y = np.zeros_like(xb[i])
            
        for j in range(stacked_size):
            fig.add_subplot(2 * batch_size, stacked_size, stacked_size * (2 * i + 1) + j + 1)
            show_label(y[j])
    return fig
    
def show_stacked(data, stacked_size=3, ordering='channel_first'):
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

def show_pair(data, labeled=True, ordering='channel_first'):
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

def compare_pair(data, labeled=True, ordering='channel_first'):
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
