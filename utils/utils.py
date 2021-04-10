import os
import numpy as np
import cv2
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
from torch.nn import init
import time
import matplotlib.pyplot as plt


def resize_and_crop_image(input_file, output_side_length, greyscale=False):
    img = cv2.imread(input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if not greyscale else cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    height_offset = (new_height - output_side_length) // 2
    width_offset = (new_width - output_side_length) // 2
    cropped_img = resized_img[height_offset:height_offset + output_side_length,
                              width_offset:width_offset + output_side_length]
    assert cropped_img.shape[:2] == (output_side_length, output_side_length)
    return cropped_img


def denormalize_minus1_1(data):
    de_data = 255*(data + 1)/2.
    return de_data.astype(np.uint8)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def save_roc_pr_curve_data(scores, labels, file_path):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'mnist':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'svhn':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    }

    return ind_to_name[dataset_name][index]


def save_model(net, model_root, filename):
    """Save trained model."""
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    torch.save(net.state_dict(), os.path.join(model_root, filename))
    print("save trained model to: {}".format(os.path.join(model_root, filename)))


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if m.affine:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def visualize_tsne_points(tsne, labels, fpath):
    class_to_idx = {
        'anomaly': 0,
        'normal': 1
    }
    colors_per_class = {
        'anomaly' : [254, 202, 87],
        'normal' : [255, 107, 107]
}
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data

        indices = [i for i, l in enumerate(labels) if l == class_to_idx[label]]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label, s=1)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    #plt.show()
    plt.savefig(fpath)
