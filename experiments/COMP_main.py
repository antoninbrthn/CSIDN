'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Main implementation of Lq, Forward and MAE models for real-world datasets SVHN and CIFAR10
'''

import pandas as pd
import os
import datetime
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

import utils.data_utils as data_utils
import utils.noise_utils as noise_utils
from utils.misc_utils import *
import models.models_pytorch as models_pytorch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files',
                    default='results/')
parser.add_argument('--import_data_path', type=str, help='dir to import dataset',
                    default='Toy_datasets/data/')
parser.add_argument('--dataset', type=str, help='MNIST, CIFAR10', default='CIFAR10')
parser.add_argument('--model', type=str, help='cnn, mlp, large_cnn', default='mlp')
parser.add_argument('--noisy_model', type=str, help='cnn, mlp, large_cnn', default='mlp')
parser.add_argument('--comp_model', type=str, help='MAE, F, LQ, CCE', default='F')
parser.add_argument('--nb_epoch', type=int, default=10)
parser.add_argument('--warm_start', type=int, default=5)
parser.add_argument('--noisy_model_epochs', type=int, default=0)
parser.add_argument('--n_features', type=int, help="Nb of feature per convolution for cnn",
                    default=128)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--mom_decay_start', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=1,
                    help='how many subprocesses to use for data loading')

args = parser.parse_args()

val_size = 0.2
n_class = 10
input_channel = 1
random_seed = args.seed
eps = 1e-2
dummy_labels = False

momentum_decay = True
momentum_decay_epoch = args.mom_decay_start

torch.manual_seed(random_seed)

if args.dataset == "MNIST":
    print("Using MNIST dataset")
    input_size = 28 * 28
    n_class = 10
    input_channel = 1

elif args.dataset == "CIFAR10":
    print("Using CIFAR10 dataset")
    input_size = 32 * 32
    n_class = 10
    input_channel = 3

elif args.dataset == "SVHN":
    print("Using SVHN dataset")
    input_size = 32 * 32
    n_class = 10
    input_channel = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device : {}".format(device))

args = {"n_class": n_class,
        "dataset": args.dataset,
        "result_dir": args.result_dir,
        "input_size": input_size,
        "input_channel": input_channel,
        "n_features": args.n_features,
        "lr": args.lr,
        "bs": args.bs,
        "nb_epoch": args.nb_epoch,
        "print_freq": args.print_freq,
        "import_data_path": args.import_data_path,
        "model": args.model,
        "noisy_model_epochs": args.noisy_model_epochs,
        "dummy_labels": dummy_labels,
        "device": device,
        "warm_start": args.warm_start,
        "noisy_model": args.noisy_model,
        "comp_model": args.comp_model
        }

# File to export results during training
save_dir = args["result_dir"]
os.makedirs(save_dir, exist_ok=True)
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
model_str = args["dataset"] + '_' + args["comp_model"] + nowTime

txtfile = os.path.join(save_dir,model_str + ".txt")
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

# Set constants
nb_epoch = args["nb_epoch"]
warm_start = args["warm_start"]

# LR scheduler
lr_schedule_bool = True
LR_epoch_step = momentum_decay_epoch
LR_gamma = 0.1


# Import data
def import_data(path, val_size, dummify=True,
                shuffle=True):
    global len_train, len_test
    x_all = np.load("{}/{}/x".format(path, args["dataset"]))
    y_all = np.load("{}/{}/y_true".format(path, args["dataset"]))
    y_pred_all = np.load("{}/{}/y_noisy".format(path, args["dataset"]))
    r_all = np.load("{}/{}/r".format(path, args["dataset"]))

    if args["dataset"] == "MNIST":
        x_all = x_all.reshape(-1, 1, 28, 28)
    if dummify:
        y_pred_all = get_dummies(y_pred_all)
        y_all = get_dummies(y_all)

    # Create train and test set
    num_train = len(y_all)
    indices = list(range(num_train))
    val_split = int(np.floor(val_size * num_train))

    if shuffle:  # Shuffle
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, test_idx = indices[val_split:], indices[:val_split]
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    y_noisy = y_pred_all[train_idx]
    r_train = r_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    y_noisy_test = y_pred_all[test_idx]
    r_test = r_all[test_idx]

    len_train, len_test = x_train.shape[0], x_test.shape[0]

    data = data_utils.Data(
        x_train, y_train, y_noisy, r_train, x_test, y_test, y_noisy_test, r_test
    )
    return data


# Get loaders
def get_loaders(data, bs=args["bs"]):
    train_loader, test_loader = data_utils.get_loader(data.x_train,
                                                      data.y_train,
                                                      data.y_noisy,
                                                      data.r_train,
                                                      data.x_test,
                                                      data.y_test,
                                                      bs=bs)
    return train_loader, test_loader


def train(train_loader, epoch, model, optimizer, criterion):
    print('Training %s...' % model_str)

    train_total = 0
    train_correct = 0
    total_loss = 0
    for i, (images, labels_clean, labels_noisy, _) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels_clean = Variable(labels_clean).to(device)
        labels_noisy = Variable(labels_noisy).to(device)

        # Forward + Backward + Optimize
        logits = model(images)
        output = F.softmax(logits, dim=1)
        predicted = output.argmax(dim=1)
        if len(labels_noisy.data.size()) == 2:
            true = torch.argmax(labels_noisy.data, dim=1)
        else:
            true = labels_noisy.data
        correct = (predicted == true).sum().item()
        acc = correct / labels_noisy.size(0)
        train_total += labels_noisy.size(0)
        train_correct += correct

        # Compute loss and update weights
        loss = criterion(output, labels_noisy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()

        if (i + 1) % args["print_freq"] == 0:
            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                % (
                epoch + 1, args["nb_epoch"], i + 1, len_train // args["bs"], acc, loss.data.item())
            )
    train_acc = float(train_correct) / float(train_total)
    return total_loss, train_acc


# Evaluate the Model
def evaluate(test_loader, model):
    print('Evaluating %s...' % model_str)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        logits = model(images)
        output = F.softmax(logits, dim=1)
        predicted = output.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * float(correct) / float(total)
    return acc


def main():
    global len_train, len_test

    # Import data
    data = import_data(path=args["import_data_path"], val_size=val_size, dummify=dummy_labels)
    x_train = data.x_train
    y_train = data.y_train
    y_noisy = data.y_noisy
    r_train = data.r_train
    x_test = data.x_test
    y_test = data.y_test

    # Get loaders
    train_loader, test_loader = get_loaders(data, bs=args["bs"])

    # Compute S
    S = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            if i != j:
                if len(y_noisy.shape) == 2:
                    S[i, j] = (to_int(y_noisy)[to_int(y_train) == j] == i).mean() / \
                              ((to_int(y_noisy)[to_int(y_train) == j] != j).mean())
                else:
                    S[i, j] = (y_noisy[y_train == j] == i).mean() / \
                              ((y_noisy[y_train == j] != j).mean())

    print("Switch matrix: \n", S)

    # Compute mu by average
    if len(y_noisy.shape) == 2:
        mus = np.array([(r_train[to_int(y_noisy) == i]).mean() for i in range(n_class)])
    else:
        mus = np.array([(r_train[y_noisy == i]).mean() for i in range(n_class)])
    print("Mean diagonal : \n", mus)

    # compute weights against class imbalance
    if dummy_labels:
        count = pd.Series(to_int(y_noisy)).value_counts()
    else:
        count = pd.Series(y_noisy).value_counts()
    total = len(y_noisy)

    print(f'Building main model: {args["model"]}...')
    if args["model"] == "cnn":
        model = models_pytorch.base_cnn_mnist(args)
    elif args["model"] == "mlp":
        model = models_pytorch.base_model_mnist(args)
    elif args["model"] == "large_cnn":
        model = models_pytorch.large_CNN(args)
    elif args["model"] == "resnet18":
        model = models_pytorch.resnet18(args)
    elif args["model"] == "resnet50":
        model = models_pytorch.resnet50(args)
    model.to(device)
    print(model.parameters)

    if args["comp_model"] == "F":
        T = noise_utils.S_to_T(S, mus=mus, n_class=args["n_class"])
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        criterion = models_pytorch.loss_forward(T)
    elif args["comp_model"] == "MAE":
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        criterion = models_pytorch.L1loss()
    elif args["comp_model"] == "LQ":
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        criterion = models_pytorch.loss_Lq(0.7)
    elif args["comp_model"] == "CCE":
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        criterion = F.cross_entropy

    else:
        raise Exception("Invalid model name.")

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc_loss train_acc test_acc \n')

    # LR scheduler
    if lr_schedule_bool:
        scheduler = StepLR(optimizer, step_size=LR_epoch_step, gamma=LR_gamma)

    # Begin training
    epoch = 0
    train_acc = 0
    # evaluate models with random weights
    test_acc = evaluate(test_loader, model)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %%' % (
    epoch + 1, args["nb_epoch"], len_test, test_acc))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': ' + str(train_acc) + ' ' + str(test_acc) + "\n")

    ########### Training ###########
    for epoch in range(nb_epoch):
        model.train()

        if epoch >= warm_start:
            print("Using dynamic beta")

        if lr_schedule_bool:
            # Decay Learning Rate
            scheduler.step()
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())

        # Train the network
        train_loss, train_acc = train(train_loader, epoch, model, optimizer, criterion)

        # Test
        test_acc = evaluate(test_loader, model)

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %% ' % (
                epoch + 1, args["nb_epoch"], len_test, test_acc))
        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ': ' + str(train_loss) + ' ' + str(train_acc) + ' ' + str(
                    test_acc) + "\n")


if __name__ == '__main__':
    main()
