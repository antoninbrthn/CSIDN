"""
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Main implementation of ILFC for synthetic datasets
"""

import os
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

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument(
    "--result_dir", type=str, help="dir to save result txt files", default="results/"
)
parser.add_argument(
    "--import_data_path",
    type=str,
    help="dir to import dataset",
    default="Toy_datasets/data/",
)
parser.add_argument("--dataset", type=str, help="MNIST, CIFAR10", default="MNIST")
parser.add_argument("--model", type=str, help="cnn, mlp, large_cnn", default="mlp")
parser.add_argument(
    "--noisy_model", type=str, help="cnn, mlp, large_cnn", default="mlp"
)
parser.add_argument(
    "--nb_epoch",
    type=int,
    default=10,
    help="Number of epochs for training the main classifier",
)
parser.add_argument(
    "--warm_start", type=int, default=5, help="Number of epochs before updating mus"
)
parser.add_argument("--noisy_model_epochs", type=int, default=10)
parser.add_argument(
    "--n_features", type=int, help="Nb of feature per convolution for cnn", default=4
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument(
    "--print_freq", type=int, default=100, help="Frequency of info during training"
)
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument("--n_iter", type=int, default=1)
parser.add_argument("--noise_type", type=str, help="sym, flip", default="sym")

# Manually set arguments
args = parser.parse_args()
args.dataset = "concentric"
args.import_data_path = "Synthetic/data/"

val_size = 0.2
n_class = 3
input_channel = 1
random_seed = args.seed
# Trim ratio
eps = 1e-2
dummy_labels = False
input_size = 2
n_iter = args.n_iter


torch.manual_seed(random_seed)

mus_tab = np.arange(0.2, 0.50, 0.05)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device : {}".format(device))

args = {
    "n_class": n_class,
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
    "noise_type": args.noise_type,
}

dataset_name = args["dataset"]
model_str = dataset_name + "_ILFC"

# Constants
nb_epoch = args["nb_epoch"]

# Iterative process params
warm_start = args["warm_start"]
squeeze_strenght = 0.5

# LR scheduler
lr_schedule_bool = False
LR_epoch_step = 40
LR_gamma = 0.1


# Import data
def import_data(path, dataset_name, val_size, dummify=True, shuffle=True):
    global len_train, len_test
    x_all = np.load("{}/{}/x".format(path, dataset_name))
    y_all = np.load("{}/{}/y_true".format(path, dataset_name))
    y_pred_all = np.load("{}/{}/y_noisy".format(path, dataset_name))
    r_all = np.load("{}/{}/r".format(path, dataset_name))

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
    train_loader, test_loader = data_utils.get_loader(
        data.x_train,
        data.y_train,
        data.y_noisy,
        data.r_train,
        data.x_test,
        data.y_test,
        bs=bs,
    )
    return train_loader, test_loader


def train(train_loader, epoch, model, noisy_model, optimizer, criterion):
    global save_dir, model_str
    print("Training %s..." % model_str)

    train_total = 0
    train_correct = 0
    total_loss = 0
    for i, (images, labels_clean, labels_noisy, r) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels_clean = Variable(labels_clean).to(device)
        labels_noisy = Variable(labels_noisy).to(device)
        r = Variable(r).to(device)

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

        if epoch >= warm_start:
            # Compute betas
            if dummy_labels:
                beta_temp = (noisy_model(images) * labels_noisy).max(dim=1)[0] / (
                    eps + (model(images) * labels_noisy).max(dim=1)[0]
                )
            else:
                noisy_dummies = (
                    torch.FloatTensor(args["bs"], args["n_class"])
                    .to(device)
                    .zero_()
                    .scatter_(1, labels_noisy.view(-1, 1), 1)
                    .to(device)
                )
                beta_temp = (noisy_model(images) * noisy_dummies).max(dim=1)[0] / (
                    eps + (model(images) * noisy_dummies).max(dim=1)[0]
                )

            beta = beta_temp
            beta = beta.detach()
        else:
            beta = torch.ones_like(r)
        beta = beta.to(device)

        # Compute loss and update weights
        loss = criterion(output, labels_noisy, r, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()

        if (i + 1) % args["print_freq"] == 0:
            print(
                "Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f"
                % (
                    epoch + 1,
                    args["nb_epoch"],
                    i + 1,
                    len_train // args["bs"],
                    acc,
                    loss.data.item(),
                )
            )
    train_acc = float(train_correct) / float(train_total)
    return total_loss, train_acc


# Evaluate the model
def evaluate(test_loader, model):
    global save_dir, model_str
    print("Evaluating %s..." % model_str)

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
    global save_dir, model_str, len_train, len_test

    def main_training(mu, iter):
        print("Beginning noise level {}".format(mu))
        # File to export results during training
        save_dir = args["result_dir"] + args["dataset"] + "/ILFC/{}/".format(str(iter))
        if not os.path.exists(save_dir):
            os.system("mkdir -p %s" % save_dir)

        dataset_name = args["dataset"] + "_%s-%.2f" % (args["noise_type"], mu)
        model_str = dataset_name + "_ILFC"
        txtfile = save_dir + "/" + model_str + ".txt"
        nowTime = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if os.path.exists(txtfile):
            if not os.path.exists(save_dir + "junk/"):
                os.system("mkdir -p %s" % save_dir + "junk/")
            os.system("mv %s %s" % (txtfile, "junk/" + txtfile + ".bak-%s" % nowTime))

        # Import data
        data = import_data(
            path=args["import_data_path"],
            dataset_name=dataset_name,
            val_size=val_size,
            dummify=dummy_labels,
        )
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
                        S[i, j] = (
                            to_int(y_noisy)[to_int(y_train) == j] == i
                        ).mean() / ((to_int(y_noisy)[to_int(y_train) == j] != j).mean())
                    else:
                        S[i, j] = (y_noisy[y_train == j] == i).mean() / (
                            (y_noisy[y_train == j] != j).mean()
                        )

        print("Switch matrix: \n", S)

        # Compute mu by average
        if len(y_noisy.shape) == 2:
            mus = np.array(
                [(r_train[to_int(y_noisy) == i]).mean() for i in range(n_class)]
            )
        else:
            mus = np.array([(r_train[y_noisy == i]).mean() for i in range(n_class)])
        print("Mean diagonal : \n", mus)

        # Compute class-noise-distributions' quantiles
        quantiles = get_noise_quantiles(r_train, y_train, args, dummy=dummy_labels)

        # Define models
        print(
            "Training naive noisy classifier for {} epochs".format(
                args["noisy_model_epochs"]
            )
        )

        noisy_model_func = models_pytorch.base_model
        noisy_model = models_pytorch.train_naive_nn(
            noisy_model_func,
            train_loader,
            test_loader,
            args,
            nb_epoch=args["noisy_model_epochs"],
            lr=args["lr"],
            x_test=x_test,
            y_test=y_test,
            device=device,
        )
        noisy_model.to(device)
        print("Done")

        print("Building main model...")

        model = models_pytorch.base_model(args)
        model.to(device)
        print(model.parameters)
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        criterion = models_pytorch.loss_general_forward_iterative(
            S, quantiles, args, use="mean", debug=False
        )

        with open(txtfile, "a") as myfile:
            myfile.write("epoch: train_acc_loss train_acc test_acc \n")

        # LR scheduler
        if lr_schedule_bool:
            scheduler = StepLR(optimizer, step_size=LR_epoch_step, gamma=LR_gamma)

        # Begin training
        epoch = 0
        train_acc = 0
        # Evaluate models with random weights
        test_acc = evaluate(test_loader, model)
        print(
            "Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %%"
            % (epoch + 1, args["nb_epoch"], len_test, test_acc)
        )
        # Save results
        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ": " + str(train_acc) + " " + str(test_acc) + "\n"
            )

        # Main training loop
        for epoch in range(nb_epoch):
            model.train()

            if epoch >= warm_start:
                print("Using dynamic beta")

            if lr_schedule_bool:
                # Decay Learning Rate
                scheduler.step()
                print("Epoch:", epoch, "LR:", scheduler.get_lr())

            # Train the network
            train_loss, train_acc = train(
                train_loader, epoch, model, noisy_model, optimizer, criterion
            )

            # Test
            test_acc = evaluate(test_loader, model)

            print(
                "Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %% "
                % (epoch + 1, args["nb_epoch"], len_test, test_acc)
            )
            with open(txtfile, "a") as myfile:
                myfile.write(
                    str(int(epoch))
                    + ": "
                    + str(train_loss)
                    + " "
                    + str(train_acc)
                    + " "
                    + str(test_acc)
                    + "\n"
                )

        plot_contour(
            model, title="ILFC - Average noise : {}%".format(str(round(100 * mu, 2)))
        )
        plt.savefig("figures/synthetic-ILFC-%d.png" % (100 * mu))

    for iter in range(n_iter):
        for mus in mus_tab:
            main_training(mus, iter)


if __name__ == "__main__":
    main()
