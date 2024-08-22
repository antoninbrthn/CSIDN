"""
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Main implementation of Lq, Forward and MAE models for experiment on Clothing1M.
"""

import datetime
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

import utils.noise_utils as noise_utils
from utils.clothing_utils import *
from utils.misc_utils import *
import models.models_pytorch as models_pytorch
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument(
    "--result_dir", type=str, help="dir to save result txt files", default="results/"
)
parser.add_argument(
    "--import_data_path",
    type=str,
    help="dir to import dataset",
    default="Toy_datasets/data/",
)
parser.add_argument(
    "--model",
    type=str,
    help="cnn, mlp, large_cnn, resnet18, resnet50",
    default="resnet18",
)
parser.add_argument("--nb_epoch", type=int, default=10)
parser.add_argument("--comp_model", type=str, help="MAE, F, LQ, DMI, CCE", default="F")
parser.add_argument("--warm_start", type=int, default=5)
parser.add_argument(
    "--n_features", type=int, help="Nb of feature per convolution for cnn", default=128
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument(
    "--mom_decay_start",
    type=int,
    help="rate of decrease of the learning rate",
    default=40,
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="how many subprocesses to use for " "data loading",
)
parser.add_argument(
    "--train_limit",
    type=int,
    default=100000,
    help="max number of batches to " "train on at each epoch",
)
parser.add_argument(
    "--model_export", type=str, default="", help="dir to export model after training"
)
parser.add_argument(
    "--model_import", type=str, default="", help="path to import model from"
)
parser.add_argument("--og_labels", type=str2bool, default=False)
parser.add_argument("--og_dataset", type=str2bool, default=False)
parser.add_argument("--optim", type=str, default="Adam", help="Adam, SGD")

args = parser.parse_args()

val_size = 0.2
input_size = 256
n_class = 14
input_channel = 3
eps = 1e-2
dummy_labels = False

if args.seed == 0:
    random_seed = np.random.randint(1000)
else:
    random_seed = args.seed

momentum_decay = True
momentum_decay_epoch = args.mom_decay_start

torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device : {}".format(device))

args = {
    "n_class": n_class,
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
    "comp_model": args.comp_model,
    "dummy_labels": dummy_labels,
    "device": device,
    "train_limit": args.train_limit,
    "model_export": args.model_export,
    "model_import": args.model_import,
    "og_labels": args.og_labels,
    "og_dataset": args.og_dataset,
    "optim": args.optim,
}

print("Arguments : ", args)

# File to export results during training
save_dir = os.path.join(args["result_dir"], "clothing", args["comp_model"])
os.makedirs(save_dir, exist_ok=True)
if args["model_export"] != "":
    os.makedirs(args["model_export"], exist_ok=True)
model_str = f'clothing_{args["comp_model"]}'

nowTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
txtfile = save_dir + "/" + model_str + "{}.txt".format(nowTime)

# Set constants
nb_epoch = args["nb_epoch"]

# LR scheduler
lr_schedule_bool = True
LR_epoch_step = momentum_decay_epoch
LR_gamma = 0.1


def train(train_loader, epoch, model, optimizer, criterion, train_limit=None):
    if train_limit is None:
        train_limit = len(train_loader)

    print("Training %s..." % model_str)

    train_total = 0
    train_correct = 0
    total_loss = 0
    for i, (images, labels_noisy, r) in enumerate(train_loader):
        if i > train_limit:
            break
        images = Variable(images).to(device)
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
                "Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f"
                % (epoch + 1, args["nb_epoch"], i + 1, len_train, acc, loss.data.item())
            )

    train_acc = float(train_correct) / float(train_total)
    return total_loss, train_acc


# Evaluate the model
def evaluate(test_loader, model):
    print("Evaluating %s..." % model_str)

    correct = 0
    total = 0
    for images, labels, _ in test_loader:
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
    data_path = args["import_data_path"]
    # Import data
    print("Importing testset")
    testset = Clothing1M(
        path=data_path,
        mode="clean_test",
        dataset_type="drive",
        transform=test_transform,
    )
    if args["og_dataset"]:
        print("Importing original Clothing1M validset")
        validset = Clothing1M(
            path=data_path,
            mode="clean_val",
            dataset_type="drive",
            transform=test_transform,
        )
        print("Importing original Clothing1M trainset")
        noisyset = Clothing1M(
            path=data_path,
            mode="noisy_train",
            dataset_type="drive",
            transform=train_transform,
        )
    else:
        print("Importing Clothing1M validset")
        validset = Clothing1M_confidence(
            path=data_path,
            fn="valid_set_labels.txt",
            transform=test_transform,
            og_labels=False,
        )
        print("Importing Clothing1M confidence scored (CSIDN) trainset")
        noisyset = Clothing1M_confidence(
            path=data_path,
            fn="train_set_labels.txt",
            transform=train_transform,
            og_labels=args["og_labels"],
        )

    train_loader = torch.utils.data.DataLoader(
        noisyset, batch_size=args["bs"], shuffle=True, num_workers=8, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args["bs"], shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args["bs"], shuffle=True, num_workers=8, pin_memory=True
    )

    len_train, len_test = len(train_loader), len(test_loader)

    if args["comp_model"] == "F":
        # Only need transition matrix and mus for forward
        # Full train and validation vectors
        y_val_noisy, y_val, r_val = validset.get_labels()
        y_train_noisy, y_train_true, r_train = noisyset.get_labels()

        # Compute S
        S = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                if i != j:
                    if len(y_val_noisy.shape) == 2:
                        S[i, j] = (
                            to_int(y_val_noisy)[to_int(y_val) == j] == i
                        ).mean() / (
                            (to_int(y_val_noisy)[to_int(y_val) == j] != j).mean()
                        )
                    else:
                        S[i, j] = (y_val_noisy[y_val == j] == i).mean() / (
                            (y_val_noisy[y_val == j] != j).mean()
                        )

        T = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                if len(y_val_noisy.shape) == 2:
                    T[i, j] = (to_int(y_val_noisy)[to_int(y_val) == j] == i).mean()
                else:
                    T[i, j] = (y_val_noisy[y_val == j] == i).mean()

        # Compute mu by average
        if len(y_val_noisy.shape) == 2:
            mus = np.array(
                [(r_val[to_int(y_val_noisy) == i]).mean() for i in range(n_class)]
            )
        else:
            mus = np.array([(r_val[y_val_noisy == i]).mean() for i in range(n_class)])
        print("Mean diagonal : \n", mus)

    print("Building main model...")
    if args["model"] == "resnet50":
        model = models_pytorch.resnet50(args)
    else:
        model = models_pytorch.resnet18(args)

    model.to(device)
    if len(args["model_import"]) > 0:
        try:
            model = torch.load(args["model_import"])
            print("Loaded model from: ", args["model_import"])
        except Exception as err:
            print("Unable to load model checkpoint. Error:", err)

    if args["comp_model"] == "F":
        T_mu = noise_utils.S_to_T(S, mus=mus, n_class=args["n_class"])
        print("T_mu - T matrix: \n", T_mu - T)
        print("T matrix: \n", T)
        criterion = models_pytorch.loss_forward(T)
    elif args["comp_model"] == "MAE":
        criterion = models_pytorch.L1loss()
    elif args["comp_model"] == "LQ":
        criterion = models_pytorch.loss_Lq(0.7)
    elif args["comp_model"] == "DMI":
        criterion = models_pytorch.loss_DMI()
    elif args["comp_model"] == "CCE":
        criterion = F.cross_entropy
    else:
        raise Exception("Invalid model name.")

    if args["optim"] == "Adam":
        print("Using Adam optimzer")
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    elif args["optim"] == "SGD":
        print("Using SGD optimizer")
        optimizer = optim.SGD(
            model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=1e-3
        )

    with open(txtfile, "a+") as myfile:
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
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(
            str(int(epoch)) + ": " + str(train_acc) + " " + str(test_acc) + "\n"
        )

    # Main training loop
    for epoch in range(nb_epoch):
        model.train()

        if lr_schedule_bool:
            # Decay Learning Rate
            scheduler.step()
            print("Epoch:", epoch, "LR:", scheduler.get_lr())

        if momentum_decay:
            if epoch >= momentum_decay_epoch:
                print("Updating beta1 in Adam optimizer")
                for param_group in optimizer.param_groups:
                    param_group["betas"] = (0.1, 0.999)

        # Train the network
        train_loss, train_acc = train(
            train_loader,
            epoch,
            model,
            optimizer,
            criterion,
            train_limit=args["train_limit"],
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

        if len(args["model_export"]) > 0:
            name = f"{args['comp_model']}_epoch{epoch}_acc{round(test_acc, 3)}"
            path = os.path.join(args["model_export"], name)
            print("Saving trained model at path: ", path)
            torch.save(model, path)


if __name__ == "__main__":
    main()
