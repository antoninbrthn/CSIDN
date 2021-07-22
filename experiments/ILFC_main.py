'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Main implementation of ILFC for real-world datasets SVHN and CIFAR10.
'''

import os
import pandas as pd
import datetime
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

import utils.data_utils as data_utils
from utils.misc_utils import *
import models.models_pytorch as models_pytorch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--import_data_path', type = str, help = 'dir to import dataset', default = 'Toy_datasets/data/')
parser.add_argument('--dataset', type = str, help = 'MNIST, CIFAR10, SVHN', default = 'MNIST')
parser.add_argument('--model', type = str, help = 'cnn, mlp, large_cnn', default = 'mlp')
parser.add_argument('--method', type = str, help = 'mean or quantiles', default = 'mean')
parser.add_argument('--noisy_model', type = str, help = 'cnn, mlp, large_cnn', default = 'mlp')
parser.add_argument('--nb_epoch', type = int, default = 10, help = 'Number of epochs for training the main classifier')
parser.add_argument('--warm_start', type = int, default = 5, help = 'Number of epochs before updating mus')
parser.add_argument('--noisy_model_epochs', type = int, default = 15, help = 'Number of epochs for training the naive classifier')
parser.add_argument('--n_features', type = int, help = 'Nb of feature per convolution for cnn', default = 128)
parser.add_argument('--seed', type = int, default = 0, help = 'Random seed')
parser.add_argument('--print_freq', type = int, default = 100, help = 'Frequency of info during training')
parser.add_argument('--bs', type = int, default = 128, help = 'Batch size')
parser.add_argument('--mom_decay_start', type = int, default = 40, help = 'Starting epoch of momentum decay')
parser.add_argument('--load_model_bool', type = str2bool, default = False, help = 'Load checkpoint model')
parser.add_argument('--load_model', type = str, default = ".", help = 'Path to checkpoint model')
parser.add_argument('--plot_reliab', type = str2bool, default = False, help='Plot reliability diagram')
parser.add_argument('--use_weights', type = str2bool, default = False, help = 'Correct class imbalance')
parser.add_argument('--update_mus', type = str2bool, default = True, help = 'Update mus iteratively')
parser.add_argument('--truncate_ratios', type = str2bool, default = False, help = 'Truncate ratios when estimating mus')
parser.add_argument('--truncate_factor', type = float, default = 0.5, help = 'Truncate factor')
parser.add_argument('--num_workers', type = int, default = 1, help = 'how many subprocesses to use for data loading')

args = parser.parse_args()

# Dataset constants
val_size = 0.2

if "MNIST" in args.dataset:
    print("Using MNIST dataset")
    input_size = 28 * 28
    n_class = 10
    input_channel = 1
elif "CIFAR10" in args.dataset:
    print("Using CIFAR10 dataset")
    input_size = 32 * 32
    n_class = 10
    input_channel = 3
elif "SVHN" in args.dataset:
    print("Using SVHN dataset")
    input_size = 32 * 32
    n_class = 10
    input_channel = 3
else:
    raise ValueError("Unrecognised dataset name.")

eps = 1e-2
dummy_labels = False
save_model = False
len_train, len_test = -1, -1  # to be defined during training
plot_reliab = args.plot_reliab
crop = 700
truncate_factor = args.truncate_factor
if args.truncate_ratios:
    print("Set truncation factor to %.2f" % truncate_factor)

save_model_dir = "./state/"
save_model_path = save_model_dir + "model_save"
if not os.path.exists(save_model_dir):
    os.system('mkdir -p %s' % save_model_dir)

if args.seed == 0:
    random_seed = np.random.randint(1000)
else:
    random_seed = args.seed

momentum_decay = True
momentum_decay_epoch = args.mom_decay_start

torch.manual_seed(random_seed)

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
        "method": args.method,
        "load_model_bool": args.load_model_bool,
        "load_model": args.load_model,
        "use_weights": args.use_weights,
        "truncate_ratios": args.truncate_ratios,
        "update_mus": args.update_mus,
        "noise_perturbation": args.noise_perturbation,
        "perturbation_type": args.perturbation_type
        }
print("Arguments : ", args)

# Log results during training
save_dir = args["result_dir"] + args["dataset"] + '/ILFC/'
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
model_str = args["dataset"] + '_ILFC'
txtfile = save_dir + "/" + model_str + ".txt"
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

nb_epoch = args["nb_epoch"]
warm_start = args["warm_start"]

# LR scheduler
lr_schedule_bool = True
LR_epoch_step = momentum_decay_epoch
LR_gamma = 0.1


# Import data
def import_data(path, val_size, dummify = True,
                shuffle = True):
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
def get_loaders(data, bs = args["bs"]):
    train_loader, test_loader = data_utils.get_loader(data.x_train,
                                                      data.y_train,
                                                      data.y_noisy,
                                                      data.r_train,
                                                      data.x_test,
                                                      data.y_test,
                                                      bs = bs)
    return train_loader, test_loader


def train(train_loader, epoch, model, noisy_model, optimizer, criterion):
    print('Training %s...' % model_str)

    train_total = 0
    train_correct = 0
    total_loss = 0
    tii_hat = []
    for i, (images, labels_clean, labels_noisy, r) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels_clean = Variable(labels_clean).to(device)
        labels_noisy = Variable(labels_noisy).to(device)
        r = Variable(r).to(device)

        # Forward + Backward + Optimize
        logits = model(images)
        output = F.softmax(logits, dim = 1)
        predicted = output.argmax(dim = 1)
        if len(labels_noisy.data.size()) == 2:
            true = torch.argmax(labels_noisy.data, dim = 1)
        else:
            true = labels_noisy.data
        correct = (predicted == true).sum().item()
        acc = correct / labels_noisy.size(0)
        train_total += labels_noisy.size(0)
        train_correct += correct

        if epoch >= warm_start:
            # Compute betas
            if dummy_labels:
                num = (noisy_model(images) * labels_noisy).max(dim = 1)[0]
                denum = (eps + (model(images) * labels_noisy).max(dim = 1)[0])
                beta_temp = num / denum
                beta_temp_trunc = torch.clamp_min(num, truncate_factor) / \
                                  torch.clamp_min(denum, truncate_factor)


            else:
                noisy_dummies = torch.FloatTensor(args["bs"], 10).to(device).zero_().scatter_(1,
                                                                                              labels_noisy.view(-1, 1),
                                                                                              1).to(device)
                num = (noisy_model(images) * noisy_dummies).max(dim = 1)[0]
                denum = (eps + (model(images) * noisy_dummies).max(dim = 1)[0])
                beta_temp = num / denum
                beta_temp_trunc = torch.clamp_min(num, truncate_factor) / \
                                  torch.clamp_min(denum, truncate_factor)

            if args["truncate_ratios"]:
                beta = beta_temp_trunc
            else:
                beta = beta_temp

            if i < 1:
                print("Average non-truncated : ", beta_temp.mean().item())
                print("Average truncated : ", beta_temp_trunc.mean().item())
                print("Std non-truncated : ", beta_temp.std().item())
                print("Std truncated : ", beta_temp_trunc.std().item())

            beta = beta.detach()

        else:
            beta = torch.ones_like(r)
        beta = beta.to(device)

        if args["method"] == "mean":
            new_tii = np.minimum((r * beta).cpu().detach().numpy(), 1)
            tii_hat += (list(new_tii))
        # Compute loss and update weights
        loss = criterion(output, labels_noisy, r, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()

        if (i + 1) % args["print_freq"] == 0:
            print(
                'Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                % (epoch + 1, args["nb_epoch"], i + 1, len_train // args["bs"], acc, loss.data.item())
            )

        if save_model:
            torch.save(model.state_dict(), "%s%.2f" % (save_model_path, acc))

    train_acc = float(train_correct) / float(train_total)
    return total_loss, train_acc, np.array(tii_hat)


# Evaluate the Model
def evaluate(test_loader, model):
    print('Evaluating %s...' % model_str)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        logits = model(images)
        output = F.softmax(logits, dim = 1)
        predicted = output.argmax(dim = 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * float(correct) / float(total)
    return acc


def main():
    global len_train, len_test

    # Import data
    data = import_data(path = args["import_data_path"], val_size = val_size, dummify = dummy_labels)
    x_train = data.x_train
    y_train = data.y_train
    y_noisy = data.y_noisy
    r_train = data.r_train
    x_test = data.x_test
    y_test = data.y_test

    # Perturbate confidence
    print("RTRAIN MEAN before perturbation: ", data.r_train.mean(), data.r_train.std())
    if args["perturbation_type"] == "normal":
        print("Adding Normal noise of strength %.2f" % args["noise_perturbation"])
        r_train = pertubate_noise(r_train, args["noise_perturbation"])
    elif args["perturbation_type"] == "over_conf":
        print("Adding Over_conf noise of strength %.2f" % args["noise_perturbation"])
        pertubate_noise(r_train, args["noise_perturbation"], form = "over_conf")
    data.r_train = r_train
    print("RTRAIN MEAN after perturbation: ", data.r_train.mean(), data.r_train.std())

    # Get loaders
    train_loader, test_loader = get_loaders(data, bs = args["bs"])

    # Compute S by cheating
    S = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            if i != j:
                if len(y_noisy.shape) == 2:
                    S[i, j] = (to_int(y_noisy)[to_int(y_train) == j] == i).mean() / \
                              ((to_int(y_noisy)[to_int(y_train) == j] != j).mean() + 1e-6)
                else:
                    S[i, j] = (y_noisy[y_train == j] == i).mean() / \
                              ((y_noisy[y_train == j] != j).mean() + 1e-6)

    print("Switch matrix: \n", S)

    # Compute mu by average
    if len(y_noisy.shape) == 2:
        mus = np.array([(r_train[to_int(y_noisy) == i]).mean() for i in range(n_class)])
    else:
        mus = np.array([(r_train[y_noisy == i]).mean() for i in range(n_class)])
    print("Mean diagonal : \n", mus)

    # Compute class-noise-distributions' quantiles
    quantiles = get_noise_quantiles(r_train, y_train, args, dummy = dummy_labels)

    # compute weights against class imbalance
    if args["use_weights"]:
        if dummy_labels:
            count = pd.Series(to_int(y_noisy)).value_counts()
        else:
            count = pd.Series(y_noisy).value_counts()
        total = len(y_noisy)
        weights = np.array([total / count[i] for i in range(n_class)])
        weights = torch.tensor(weights).to(device).float()
        print("Weights: ", weights)
    else:
        weights = None

    # Define models
    print("Training naive noisy {} classifier for {} epochs".format(args["noisy_model"], args["noisy_model_epochs"]))
    if args["noisy_model"] == "cnn":
        noisy_model_func = models_pytorch.base_cnn_mnist
        # input_channel = args["input_channel"],hidden_size=args["n_features"],n_outputs = n_class,dropout_rate = 0.25)
    elif args["noisy_model"] == "mlp":
        noisy_model_func = models_pytorch.base_cnn_mnist
    elif args["noisy_model"] == "large_cnn":
        noisy_model_func = models_pytorch.large_CNN
    noisy_model = models_pytorch.train_naive_nn(noisy_model_func,
                                                train_loader,
                                                test_loader,
                                                args,
                                                nb_epoch = args["noisy_model_epochs"], lr = args["lr"],
                                                x_test = x_test,
                                                y_test = y_test,
                                                device = device,
                                                weights = None)
    print(noisy_model.parameters)
    print("Done")

    if plot_reliab:
        if not os.path.exists("/figures"):
            os.system('mkdir -p %s' % "/figures")
        # Plot reliability diagram
        intervals, acc = get_reliability_diag(noisy_model.cpu()(torch.tensor(x_test[:crop])),
                                              torch.tensor(y_test[:crop]),
                                              nb_bins = 10)
        plt.bar(intervals, acc, width = 0.1)
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        plt.title('Final noisy classifier')
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.xlim([0, 1])
        plt.savefig("Calibration_noisy")
        noisy_model.to(device)

    print('Building main model...')

    if args["model"] == "cnn":
        model = models_pytorch.base_cnn_mnist(args)
        # input_channel = args["input_channel"],hidden_size=args["n_features"],n_outputs = n_class,dropout_rate = 0.25)
    elif args["model"] == "mlp":
        model = models_pytorch.base_model_mnist(args)
    elif args["model"] == "large_cnn":
        model = models_pytorch.large_CNN(args)

    model.to(device)
    print(model.parameters)
    if args["load_model_bool"]:
        print("Loading model from {}".format(args["load_model"]))
        model.load_state_dict(torch.load(args["load_model"]))

    optimizer = optim.Adam(model.parameters(), lr = args["lr"])
    criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args, use = args["method"],
                                                              debug = False, weights = weights)

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc_loss train_acc test_acc \n')

    # LR scheduler
    if lr_schedule_bool:
        scheduler = StepLR(optimizer, step_size = LR_epoch_step, gamma = LR_gamma)

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

        if momentum_decay:
            if epoch >= momentum_decay_epoch:
                print("Updating beta1 in Adam optimizer")
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (0.1, 0.999)

        # Train the network
        train_loss, train_acc, tii_hat = train(train_loader, epoch, model, noisy_model, optimizer, criterion)

        if args["method"] == "mean":
            if args["update_mus"]:
                # Update mu by average
                print("Updating mus coefficients")
                if len(y_noisy.shape) == 2:
                    mus = np.array([(tii_hat[to_int(y_noisy[:len(tii_hat)]) == i]).mean() for i in range(n_class)])
                else:
                    mus = np.array([(tii_hat[y_noisy[:len(tii_hat)] == i]).mean() for i in range(n_class)])
                criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args,
                                                                          use = args["method"], debug = False,
                                                                          weights = weights)


        elif args["method"] == "quantiles":
            print("Computing new quantiles and updating criterion")
            quantiles = get_noise_quantiles(tii_hat, y_train[:len(tii_hat)], args, dummy = dummy_labels)
            print("New quantiles :")
            print(quantiles)
            criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args,
                                                                      use = args["method"], debug = False,
                                                                      weights = weights)

        # Test
        test_acc = evaluate(test_loader, model)

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %% ' % (
                epoch + 1, args["nb_epoch"], len_test, test_acc))
        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ': ' + str(train_loss) + ' ' + str(train_acc) + ' ' + str(test_acc) + "\n")

        if plot_reliab:
            if not os.path.exists("/figures/"):
                os.system('mkdir -p %s' % "/figures/")
            # Plot reliability diagram
            intervals, acc = get_reliability_diag(model.cpu()(torch.tensor(x_test[:crop])), torch.tensor(y_test[:crop]),
                                                  nb_bins = 10)
            plt.bar(intervals, acc, width = 0.1)
            plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
            plt.title('Main classifier - Epoch %d' % (epoch))
            plt.xlabel("Confidence")
            plt.ylabel("Accuracy")
            plt.xlim([0, 1])
            plt.savefig("Calibration-e%d" % (epoch))


if __name__ == '__main__':
    main()
