import os
import argparse
import json
from datetime import datetime
import time
import socket
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models.rae import CAE
from dataset.outlier_datasets import (load_cifar10_with_outliers,
                                      load_cifar100_with_outliers,
                                      load_fashion_mnist_with_outliers,
                                      load_mnist_with_outliers,
                                      load_svhn_with_outliers,
                                      get_channels_axis,
                                      )
from dataset.dataset import trainset_pytorch, trainset_rat
from utils.misc import AverageMeter
from utils.utils import (save_roc_pr_curve_data,
                         get_class_name_from_index,
                         save_model,
                         init_weights,
                         convert_secs2time,
                         time_string,
                         visualize_tsne_points,
                         denormalize_minus1_1)

matplotlib.use('Agg')
cudnn.benchmark = True
HOST = socket.gethostname()
RESULTS_DIR = 'results/models/rae/_raw/' + \
    datetime.now().strftime('%Y-%m-%d-%H%M%S') + '-' + HOST
logger = SummaryWriter(RESULTS_DIR)


#########################
# functions
#########################
def visualize(input, output, fpath, num):
    input_img = torchvision.utils.make_grid(input[:num, :, :, :], nrow=8)
    output_img = torchvision.utils.make_grid(output[:num, :, :, :], nrow=8)
    merge_img = torchvision.utils.make_grid([input_img, output_img], nrow=1)
    merge_img_de = denormalize_minus1_1(merge_img.cpu().detach().numpy())
    plt.imshow(np.transpose(merge_img_de, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.savefig(fpath)


def dec_loss_fun(rep, centroid):
    alpha = 1.0
    distance = torch.sum((rep.unsqueeze(1) - centroid) ** 2, dim=2)
    distance_norm = distance / torch.max(distance)
    q_pos = 1.0 / (1.0 + 5 * distance_norm)
    q_neg = 1 - q_pos
    q = torch.cat((q_pos, q_neg), dim=1)
    tmp = torch.sum(q, dim=0)
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p, reduction='batchmean')
    return loss, p.data.cpu().numpy()


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


def update_center_c(reps, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    c = torch.mean(reps, dim=0, keepdim=True)
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


#########################
# train and test
#########################
def train_cae(trainloader, model, class_name, testloader, y_train, device, args):
    """
    model train function.
    :param trainloader:
    :param model:
    :param class_name:
    :param testloader:
    :param y_train: numpy array, sample normal/abnormal labels, [1 1 1 1 0 0] like, original sample size.
    :param device: cpu or gpu:0/1/...
    :param args:
    :return:
    """
    global_step = 0
    losses = AverageMeter()
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(1, args.epochs + 1):
        model.train()

        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)
        print('{:3d}/{:3d} ----- {:s} {:s}'.format(epoch,
              args.epochs, time_string(), need_time))

        mse = nn.MSELoss(reduction='mean')  # default

        lr = 0.1 / pow(2, np.floor(epoch / args.lr_schedule))
        logger.add_scalar(class_name + "/lr", lr, epoch)

        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=args.weight_decay)
        else:
            optimizer = optim.Adam(
                model.parameters(), eps=1e-7, weight_decay=0.0005)
        for batch_idx, (input, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            input = input.to(device)

            _, output = model(input)
            loss = mse(input, output)
            losses.update(loss.item(), 1)

            logger.add_scalar(class_name + '/loss', losses.avg, global_step)

            global_step = global_step + 1
            loss.backward()
            optimizer.step()

        # print losses
        print('Epoch: [{} | {}], loss: {:.4f}'.format(
            epoch, args.epochs, losses.avg))

        # log images
        if epoch % args.log_img_steps == 0:
            os.makedirs(os.path.join(RESULTS_DIR, class_name), exist_ok=True)
            fpath = os.path.join(RESULTS_DIR, class_name,
                                 'pretrain_epoch_' + str(epoch) + '.png')
            visualize(input, output, fpath, num=32)

        # test while training
        if epoch % args.log_auc_steps == 0:
            rep, losses_result = test(
                testloader, model, class_name, args, device, epoch)

            centroid = torch.mean(rep, dim=0, keepdim=True)

            losses_result = losses_result - losses_result.min()
            losses_result = losses_result / (1e-8 + losses_result.max())
            scores = 1 - losses_result
            auroc_rec = roc_auc_score(y_train, scores)

            _, p = dec_loss_fun(rep, centroid)
            score_p = p[:, 0]
            auroc_dec = roc_auc_score(y_train, score_p)

            print("Epoch: [{} | {}], auroc_rec: {:.4f}; auroc_dec: {:.4f}".format(
                epoch, args.epochs, auroc_rec, auroc_dec))

            logger.add_scalar(class_name + '/auroc_rec', auroc_rec, epoch)
            logger.add_scalar(class_name + '/auroc_dec', auroc_dec, epoch)

        # time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()


def train_iae(trainloader, model, class_name, testloader, y_train, device, args):
    """
    model train function.
    :param trainloader:
    :param model:
    :param class_name:
    :param testloader:
    :param y_train: numpy array, sample normal/abnormal labels, [1 1 1 1 0 0] like, original sample size.
    :param device: cpu or gpu:0/1/...
    :param args:
    :return:
    """
    global_step = 0
    losses = AverageMeter()
    l2_losses = AverageMeter()
    svdd_losses = AverageMeter()

    start_time = time.time()
    epoch_time = AverageMeter()

    svdd_loss = torch.tensor(0, device=device)
    R = torch.tensor(0, device=device)
    c = torch.randn(256, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()

        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)
        print('{:3d}/{:3d} ----- {:s} {:s}'.format(epoch,
              args.epochs, time_string(), need_time))

        mse = nn.MSELoss(reduction='mean')  # default

        lr = 0.1 / pow(2, np.floor(epoch / args.lr_schedule))
        logger.add_scalar(class_name + "/lr", lr, epoch)

        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(), eps=1e-7, weight_decay=args.weight_decay)
        else:
            print('not implemented.')

        for batch_idx, (input, _, _) in enumerate(trainloader):
            optimizer.zero_grad()
            input = input.to(device)

            reps, output = model(input)

            if epoch > args.pretrain_epochs:
                dist = torch.sum((reps - c) ** 2, dim=1)
                scores = dist - R ** 2
                svdd_loss = args.para_lambda * (R ** 2 + (1 / args.para_nu) * torch.mean(
                    torch.max(torch.zeros_like(scores), scores)))

            l2_loss = mse(input, output)

            loss = l2_loss + svdd_loss

            l2_losses.update(l2_loss.item(), 1)
            svdd_losses.update(svdd_loss.item(), 1)
            losses.update(loss.item(), 1)

            logger.add_scalar(class_name + '/l2_loss',
                              l2_losses.avg, global_step)
            logger.add_scalar(class_name + '/svdd_loss',
                              svdd_losses.avg, global_step)
            logger.add_scalar(class_name + '/loss', losses.avg, global_step)

            logger.add_scalar(class_name + '/R', R.data, global_step)

            global_step = global_step + 1
            loss.backward()
            optimizer.step()

            # Update hypersphere radius R on mini-batch distances
            if epoch > args.pretrain_epochs:
                R.data = torch.tensor(get_radius(
                    dist, args.para_nu), device=device)

        # print losses
        print('Epoch: [{} | {}], loss: {:.4f}'.format(
            epoch, args.epochs, losses.avg))

        # log images
        if epoch % args.log_img_steps == 0:
            os.makedirs(os.path.join(RESULTS_DIR, class_name), exist_ok=True)
            fpath = os.path.join(RESULTS_DIR, class_name,
                                 'pretrain_epoch_' + str(epoch) + '.png')
            visualize(input, output, fpath, num=32)

        # test while training
        if epoch % args.log_auc_steps == 0:
            rep, losses_result = test(
                testloader, model, class_name, args, device, epoch)

            centroid = torch.mean(rep, dim=0, keepdim=True)

            losses_result = losses_result - losses_result.min()
            losses_result = losses_result / (1e-8 + losses_result.max())
            scores = 1 - losses_result
            auroc_rec = roc_auc_score(y_train, scores)

            _, p = dec_loss_fun(rep, centroid)
            score_p = p[:, 0]
            auroc_dec = roc_auc_score(y_train, score_p)

            print("Epoch: [{} | {}], auroc_rec: {:.4f}; auroc_dec: {:.4f}".format(
                epoch, args.epochs, auroc_rec, auroc_dec))

            logger.add_scalar(class_name + '/auroc_rec', auroc_rec, epoch)
            logger.add_scalar(class_name + '/auroc_dec', auroc_dec, epoch)

        # initial centroid c before pretrain finished
        if epoch == args.pretrain_epochs:
            rep, losses_result = test(
                testloader, model, class_name, args, device, epoch)
            c = update_center_c(rep)

        # time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()


def test(testloader, model, class_name, args, device, epoch):
    model.eval()
    losses = []
    reps = []
    for batch_idx, (input, labels, id) in enumerate(testloader):
        with torch.no_grad():
            input = input.to(device)

            rep, output = model(input)

            # L1 loss
            l1loss = output.sub(input).abs().view(output.size(0), -1)
            l1loss = l1loss.sum(dim=1, keepdim=False)

            # L2 loss
            #l2loss = mse(data, output)
            #l2loss = l2loss.sum(dim=(1,2,3), keepdim=False)

            # SSIM loss
            #ssimloss = ssim(data, output)

            #loss = loss + l1loss + l2loss
            loss = l1loss
        losses.append(loss.data.cpu())
        reps.append(rep)

    # log images
    # if (batch_idx + 1) % 100 == 0 and epoch == 999:
    #     inputs_img = torchvision.utils.make_grid(inputs[:32, :, :, :])
    #     inputs_transformed_img = torchvision.utils.make_grid(inputs_transformed[:32, :, :, :])
    #     outputs_img = torchvision.utils.make_grid(outputs[:32, :, :, :])
    #     merge_img = torchvision.utils.make_grid(
    #         [inputs_img, inputs_transformed_img.expand(3, -1, -1), outputs_img],
    #         nrow=1)
    #     merge_img_de = denormalize_minus1_1(merge_img.cpu().detach().numpy())
    #     plt.imshow(np.transpose(merge_img_de, (1, 2, 0)), interpolation='nearest')
    #     plt.axis('off')
    #     os.makedirs(os.path.join(MODEL_DIR, class_name), exist_ok=True)
    #     plt.savefig(os.path.join(MODEL_DIR, class_name,
    #                              'test_epoch_' + str(epoch) + '_batch_' + str(batch_idx) + '.png'))

    losses = torch.cat(losses, dim=0)
    reps = torch.cat(reps, dim=0)
    return reps, losses.numpy()


#########################
# methods
#########################
def cae(x_train, y_train, class_idx, restore, args):
    """ l2loss
    :param x_train:
    :param y_train:
    :param class_idx:
    :param restore:
    :param args:
    :return:
    """
    device = torch.device(
        "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([transforms.ToTensor(), ])
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        print("do not use data augmentation")
    elif args.augmentation == 1:
        print("Using data augmentation")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ])

    n_channels = x_train.shape[get_channels_axis()]

    class_name = get_class_name_from_index(class_idx, args.dataset)

    model = CAE(in_channels=n_channels)

    model = model.to(device)
    init_weights(model, init_type='xavier', init_gain=0.02)

    trainset = trainset_pytorch(train_data=x_train,
                                train_labels=y_train,
                                transform=transform_train)
    testset = trainset_pytorch(train_data=x_train,
                               train_labels=y_train,
                               transform=transform_test)

    trainloader = data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)

    # training
    if not restore:
        train_cae(trainloader, model, class_name,
                  testloader, y_train, device, args)
        if args.save_model == 1:
            model_file_name = '{}_cae-{}_{}.model.npz'.format(
                args.dataset, args.ratio, class_name)
            model_path = os.path.join(RESULTS_DIR, args.dataset)
            save_model(model, model_path, model_file_name)
    else:
        print("restore model from: {}".format(restore))
        model.load_state_dict(torch.load(restore))

    # testing
    reps, losses = test(testloader, model, class_name, args, device, epoch=-1)

    # AUROC based on reconstruction losses
    losses = losses - losses.min()
    losses = losses / (1e-8 + losses.max())
    scores = 1 - losses  # normal: label=1, score near 1, loss near 0

    res_file_name = '{}_cae_rec-{}_{}_{}.npz'.format(
        args.dataset, args.ratio, class_name, datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, args.dataset, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, args.dataset), exist_ok=True)
    auc_roc_rec = roc_auc_score(y_train, scores)
    print('testing result: auc_rec: {:.4f}'.format(auc_roc_rec))
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    # DEC based on reconstruction losses
    centroid = torch.mean(reps, dim=0, keepdim=True)
    _, p = dec_loss_fun(reps, centroid)
    score_p = p[:, 0]

    res_file_name = '{}_cae_dec-{}_{}_{}.npz'.format(
        args.dataset, args.ratio, class_name, datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, args.dataset, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, args.dataset), exist_ok=True)
    auc_roc_dec = roc_auc_score(y_train, score_p)
    print('testing result: auc_dec: {:.4f}'.format(auc_roc_dec))
    save_roc_pr_curve_data(score_p, y_train, res_file_path)


def iae(x_train, y_train, class_idx, restore, args):
    """ l2loss
    :param x_train:
    :param y_train:
    :param class_idx:
    :param restore:
    :param args:
    :return:
    """
    device = torch.device(
        "cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([transforms.ToTensor(), ])
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        print("Not using data augmentation")
    elif args.augmentation == 1:
        print("Using data augmentation")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ])

    n_channels = x_train.shape[get_channels_axis()]

    class_name = get_class_name_from_index(class_idx, args.dataset)

    model = CAE(in_channels=n_channels)

    model = model.to(device)
    init_weights(model, init_type='xavier', init_gain=0.02)

    trainset = trainset_pytorch(train_data=x_train,
                                train_labels=y_train,
                                transform=transform_train)
    testset = trainset_pytorch(train_data=x_train,
                               train_labels=y_train,
                               transform=transform_test)

    trainloader = data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testloader = data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False)

    # training
    if not restore:
        train_iae(trainloader, model, class_name,
                  testloader, y_train, device, args)
        if args.save_model == 1:
            model_file_name = '{}_iae-{}_{}.model.npz'.format(
                args.dataset, args.ratio, class_name)
            model_path = os.path.join(RESULTS_DIR, args.dataset)
            save_model(model, model_path, model_file_name)
    else:
        print("restore model from: {}".format(restore))
        model.load_state_dict(torch.load(restore))

    # testing
    reps, losses = test(testloader, model, class_name, args, device, epoch=-1)

    # AUROC based on reconstruction losses
    losses = losses - losses.min()
    losses = losses / (1e-8 + losses.max())
    scores = 1 - losses  # normal: label=1, score near 1, loss near 0

    res_file_name = '{}_iae_rec-{}_{}_{}.npz'.format(
        args.dataset, args.ratio, class_name, datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, args.dataset, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, args.dataset), exist_ok=True)
    auc_roc_rec = roc_auc_score(y_train, scores)
    print('testing result: auc_rec: {:.4f}'.format(auc_roc_rec))
    save_roc_pr_curve_data(scores, y_train, res_file_path)

    # DEC based on reconstruction losses
    centroid = torch.mean(reps, dim=0, keepdim=True)
    _, p = dec_loss_fun(reps, centroid)
    score_p = p[:, 0]

    res_file_name = '{}_iae_dec-{}_{}_{}.npz'.format(
        args.dataset, args.ratio, class_name, datetime.now().strftime('%Y-%m-%d-%H%M'))
    res_file_path = os.path.join(RESULTS_DIR, args.dataset, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, args.dataset), exist_ok=True)
    auc_roc_dec = roc_auc_score(y_train, score_p)
    print('testing result: auc_dec: {:.4f}'.format(auc_roc_dec))
    save_roc_pr_curve_data(score_p, y_train, res_file_path)


def main():
    parser = argparse.ArgumentParser(description='RAE experiment parameters.')
    parser.add_argument('--run_times', type=int, default=1,
                        help='how many run times, default 1 time.')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='which gpu to use, default id 0.')
    parser.add_argument('--dataset', type=str,
                        default='cifar10', help='which dataset used.')
    parser.add_argument('--normal_class', type=int, default=-1,
                        help='which class used as normal class, default all classes.')
    parser.add_argument('--ratio', type=float, default=-1,
                        help='outlier ratio used, default all ratio [0.05, ..., 0.25]')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size.')
    parser.add_argument('--epochs', type=int, default=250,
                        help='training epochs.')

    parser.add_argument('--log_img_steps', type=int,
                        default=1000, help='log_img_steps during training.')
    parser.add_argument('--log_auc_steps', type=int, default=5,
                        help='log_auc_steps during training.')
    parser.add_argument('--save_model', type=int, default=0,
                        help='whether save model, default 0, do not save.')
    parser.add_argument('--method', type=str, default='IAE',
                        help='choose which method to run, IAE or CAE.')

    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='how many pretrain steps before add svdd loss.')
    parser.add_argument('--augmentation', type=int, default=0,
                        help='turn on/off data augmentation, default off.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='SGD or ADAM (eps=1e-7, weight_decay=0.0005).')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (L2 penalty), default 0')
    parser.add_argument('--lr_schedule', type=float, default=50,
                        help='learning rate schedule parameter.')
    parser.add_argument('--para_nu', type=float,
                        default=0.1, help='deep svdd parameter.')
    parser.add_argument('--para_lambda', type=float,
                        default=5e-5, help='deep svdd parameter.')

    args = parser.parse_args()

    experiments_dict = {
        'mnist': (load_mnist_with_outliers, 'mnist', 10),
        'fashion-mnist': (load_fashion_mnist_with_outliers, 'fashion-mnist', 10),
        'cifar10': (load_cifar10_with_outliers, 'cifar10', 10),
        'cifar100': (load_cifar100_with_outliers, 'cifar100', 20),
        'svhn': (load_svhn_with_outliers, 'svhn', 10)
    }
    method_dict = {
        'RAE': iae,
        'CAE': cae
    }
    if args.ratio == -1:
        ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    else:
        ratio_list = [args.ratio]

    with open(RESULTS_DIR+'/' + '_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    n_run = args.run_times
    data_load_fn, dataset_name, class_num = experiments_dict[args.dataset]
    method = method_dict[args.method]

    for run_idx in range(n_run):
        max_sample_num = 12000
        os.makedirs(os.path.join(RESULTS_DIR, args.dataset), exist_ok=True)
        for class_idx in range(0, class_num):
            if args.normal_class == -1:
                pass
            else:
                class_idx = args.normal_class
            for ratio in ratio_list:
                args.ratio = ratio
                class_name = get_class_name_from_index(class_idx, args.dataset)
                np.random.seed(run_idx)
                x_train, y_train = data_load_fn(class_idx, ratio)

                # random sampling if the number of data is too large
                if x_train.shape[0] > max_sample_num:
                    selected = np.random.choice(
                        x_train.shape[0], max_sample_num, replace=False)
                    x_train = x_train[selected, :]
                    y_train = y_train[selected]
                print('current training dataset: {}, normal class: {}, ratio: {}.'.format(
                    dataset_name, class_name, ratio))
                method(x_train, y_train, class_idx, None, args)
            if args.normal_class == -1:
                pass
            else:
                break


if __name__ == '__main__':
    main()
