'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Main implementation of ILFC for experiment on Clothing1M.
'''

import pandas as pd
import datetime
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from utils.clothing_utils import *
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
parser.add_argument('--model', type = str, help = 'cnn, mlp, large_cnn, resnet18, resnet50', default = 'mlp')
parser.add_argument('--method', type = str, help = 'mean or quantiles', default = 'mean')
parser.add_argument('--noisy_model', type = str, help = 'cnn, mlp, large_cnn', default = 'mlp')
parser.add_argument('--nb_epoch', type = int, default = 10, help = 'Number of epochs for training the main classifier')
parser.add_argument('--warm_start', type = int, default = 5, help = 'Number of epochs before updating mus')
parser.add_argument('--noisy_model_epochs', type = int, default = 15,
                    help = 'Number of epochs for training the naive classifier')
parser.add_argument('--n_features', type = int, help = 'Nb of feature per convolution for cnn', default = 128)
parser.add_argument('--seed', type = int, default = 0, help = 'Random seed')
parser.add_argument('--print_freq', type = int, default = 100, help = 'Frequency of info during training')
parser.add_argument('--bs', type = int, default = 128, help = 'Batch size')
parser.add_argument('--mom_decay_start', type = int, default = 40, help = 'Starting epoch of momentum decay')
parser.add_argument('--plot_reliab', type = str2bool, default = False, help = 'Plot reliability diagram')
parser.add_argument('--use_weights', type = str2bool, default = False, help = 'Correct class imbalance')
parser.add_argument('--update_mus', type = str2bool, default = True, help = 'Update mus iteratively')
parser.add_argument('--truncate_ratios', type = str2bool, default = False, help = 'Truncate ratios when estimating mus')
parser.add_argument('--truncate_factor', type = float, default = 0.5, help = 'Truncate factor')
parser.add_argument('--num_workers', type = int, default = 1, help = 'how many subprocesses to use for data loading')
parser.add_argument('--train_limit', type = int, default = 100000, help = 'max number of batches to '
                                                                          'train on at each epoch')
parser.add_argument('--noisy_model_export', type = str, default = '', help = 'dir to export noisy model after training')
parser.add_argument('--noisy_model_import', type = str, default = '', help = 'path to import noisy model from')
parser.add_argument('--model_export', type = str, default = '', help = 'dir to export model after training')
parser.add_argument('--model_import', type = str, default = '', help = 'path to import model from')
parser.add_argument('--og_labels', type = str2bool, default = False, help = 'Run with original Clothing1M labels')
parser.add_argument('--noisy_temp', type = float, default = 1., help = 'Temperature for calibrating noisy model')
parser.add_argument('--optim', type = str, default = 'Adam', help = 'Adam, SGD')

args = parser.parse_args()

val_size = 0.2
input_size = 256
n_class = 14
input_channel = 3
eps = 1e-2
dummy_labels = False
save_model = False
plot_reliab = args.plot_reliab
truncate_factor = args.truncate_factor
if args.truncate_ratios:
    print("Set truncation factor to %.2f" % truncate_factor)

if args.seed == 0:
    random_seed = np.random.randint(1000)
else:
    random_seed = args.seed  # (86%)

momentum_decay = True
momentum_decay_epoch = args.mom_decay_start

torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device : {}".format(device))

args = {"n_class": n_class,
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
        "use_weights": args.use_weights,
        "truncate_ratios": args.truncate_ratios,
        "update_mus": args.update_mus,
        "train_limit": args.train_limit,
        "noisy_model_export": args.noisy_model_export,
        "noisy_model_import": args.noisy_model_import,
        "model_export": args.model_export,
        "model_import": args.model_import,
        "og_labels": args.og_labels,
        "noisy_temp": args.noisy_temp,
        "optim": args.optim,
        }
print("Arguments : ", args)

# File to export results during training
save_dir = os.path.join(args["result_dir"], 'clothing', 'ILFC')
os.makedirs(save_dir, exist_ok=True)
if args["model_export"] != "":
    os.makedirs(args["model_export"], exist_ok=True)
if args["noisy_model_export"] != "":
    os.makedirs(args["noisy_model_export"], exist_ok=True)
model_str = 'clothing_ILFC'

nowTime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
txtfile = save_dir + "/" + model_str + "{}.txt".format(nowTime)
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

# Set constants
nb_epoch = args["nb_epoch"]
warm_start = args["warm_start"]

# LR scheduler
lr_schedule_bool = True
LR_epoch_step = momentum_decay_epoch
LR_gamma = 0.1


def train(train_loader, epoch, model, noisy_model, optimizer, criterion, train_limit = None):
    model.train()
    if train_limit is None:
        train_limit = len(train_loader)

    print('Training %s...' % model_str)

    train_total = 0
    train_correct = 0
    total_loss = 0
    tii_hat = []
    for i, (images, labels_noisy, r) in enumerate(train_loader):
        if i > train_limit:
            break
        images = Variable(images).to(device)
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
            noisy_model.to(device)
            # Compute betas
            if dummy_labels:
                num = (noisy_model(images) * labels_noisy).max(dim = 1)[0]
                denum = (eps + (model(images) * labels_noisy).max(dim = 1)[0])
                beta_temp = num / denum
                beta_temp_trunc = torch.clamp_min(num, truncate_factor) / \
                                  torch.clamp_min(denum, truncate_factor)


            else:
                noisy_dummies = torch.FloatTensor(args["bs"], n_class).to(device).zero_().scatter_(1,
                                                                                                   labels_noisy.view(
                                                                                                       -1,
                                                                                                       1),
                                                                                                   1).to(
                    device)
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
                % (
                    epoch + 1, args["nb_epoch"], i + 1, len_train, acc, loss.data.item())
            )

        if save_model:
            torch.save(model.state_dict(), "%s%.2f" % (save_model_path, acc))

    train_acc = float(train_correct) / float(train_total)
    return total_loss, train_acc, np.array(tii_hat)


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()
    print('Evaluating %s...' % model_str)

    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        with torch.no_grad():
            logits = model(images)
        output = F.softmax(logits, dim = 1)
        predicted = output.argmax(dim = 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * float(correct) / float(total)
    return acc


def main():
    global len_train, len_test
    data_path = args['import_data_path']
    # Import data
    print("Importing testset")
    testset = Clothing1M(path = data_path, mode = 'clean_test', dataset_type = 'drive',
                         transform = test_transform)
    print("Importing validset")
    validset = Clothing1M_confidence(path = data_path, fn = "valid_export_jan21.txt",
                                     transform = test_transform, og_labels = False)
    print("Importing trainset")
    noisyset = Clothing1M_confidence(path = data_path, fn = "noisy_export_jan21.txt",
                                     transform = train_transform, og_labels = args["og_labels"])

    train_loader = torch.utils.data.DataLoader(noisyset,
                                               batch_size = args['bs'],
                                               shuffle = True,
                                               num_workers = 8,
                                               pin_memory = True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size = args['bs'],
                                               shuffle = True,
                                               num_workers = 8,
                                               pin_memory = True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size = args['bs'],
                                              shuffle = False,
                                              num_workers = 8,
                                              pin_memory = False)

    len_train, len_test = len(train_loader), len(test_loader)

    # Full train and validation vectors
    y_val_noisy, y_val, r_val = validset.get_labels()
    y_train_noisy, y_train_true, r_train = noisyset.get_labels()
    # Compute S
    S = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            if i != j:
                if len(y_val_noisy.shape) == 2:
                    S[i, j] = (to_int(y_val_noisy)[to_int(y_val) == j] == i).mean() / \
                              ((to_int(y_val_noisy)[to_int(y_val) == j] != j).mean())
                else:
                    S[i, j] = (y_val_noisy[y_val == j] == i).mean() / \
                              ((y_val_noisy[y_val == j] != j).mean())

    print("Switch matrix: \n", S)

    # Compute mu by average
    if len(y_val_noisy.shape) == 2:
        mus = np.array([(r_val[to_int(y_val_noisy) == i]).mean() for i in range(n_class)])
    else:
        mus = np.array([(r_val[y_val_noisy == i]).mean() for i in range(n_class)])
    print("Mean diagonal : \n", mus)

    # Compute class-noise-distributions' quantiles
    quantiles = get_noise_quantiles(r_val, y_val, args, dummy = dummy_labels)
    print("Quantiles means:")
    print(quantiles[:, 5])

    # compute weights against class imbalance
    if args["use_weights"]:
        if dummy_labels:
            count = pd.Series(to_int(y_train_noisy)).value_counts()
        else:
            count = pd.Series(y_train_noisy).value_counts()
        total = len(y_train_noisy)
        weights = np.array([total / count[i] for i in range(n_class)])
        weights = torch.tensor(weights).to(device).float()
        print("Weights: ", weights)
    else:
        weights = None

    # Define models
    print("Training naive noisy {} classifier for {} epochs".format(args["noisy_model"],
                                                                    args["noisy_model_epochs"]))
    if args["noisy_model"] == "cnn":
        noisy_model_func = models_pytorch.base_cnn_mnist
    elif args["noisy_model"] == "mlp":
        noisy_model_func = models_pytorch.base_cnn_mnist
    elif args["noisy_model"] == "large_cnn":
        noisy_model_func = models_pytorch.large_CNN
    elif args["noisy_model"] == "resnet18":
        noisy_model_func = models_pytorch.resnet18_temp
    elif args["noisy_model"] == "resnet50":
        noisy_model_func = models_pytorch.resnet50
    else:
        raise ValueError("Unrecognised name for noisy model architecture.")
    noisy_model = models_pytorch.train_naive_nn(noisy_model_func,
                                                train_loader,
                                                valid_loader,
                                                args,
                                                nb_epoch = args["noisy_model_epochs"],
                                                lr = args["lr"],
                                                device = device,
                                                weights = None,
                                                show_ece = False,
                                                clothing_exp = True,
                                                train_limit = args['train_limit'],
                                                export_path = args['noisy_model_export'],
                                                import_path = args['noisy_model_import'],
                                                T = args["noisy_temp"])
    print(noisy_model.parameters)
    print("Done")

    print("Switching noisy model to cpu until warm start is over.")
    noisy_model.to('cpu')

    print('Building main model...')

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
    else:
        raise ValueError("Unrecognised name for model architecture.")

    model.to(device)
    if len(args["model_import"]) > 0:
        try:
            model = torch.load(args["model_import"])
            print("Loaded model from: ", args["model_import"])
        except Exception as err:
            print("Unable to load model checkpoint. Error:", err)

    if args["optim"] == 'Adam':
        print('Using Adam optimzer')
        optimizer = optim.Adam(model.parameters(), lr = args["lr"])
    elif args["optim"] == 'SGD':
        print('Using SGD optimizer')
        optimizer = optim.SGD(model.parameters(),
                              lr = args["lr"],
                              momentum = 0.9,
                              weight_decay = 1e-3)
    criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args,
                                                              use = args["method"], debug = False,
                                                              weights = weights)

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
        train_loss, train_acc, tii_hat = train(train_loader, epoch, model, noisy_model, optimizer,
                                               criterion, train_limit = args['train_limit'])

        if args["method"] == "mean":
            if args["update_mus"]:
                # Update mu by average
                print("Updating mus coefficients")
                if len(y_train_noisy.shape) == 2:
                    mus = np.array([(tii_hat[to_int(y_train_noisy[:len(tii_hat)]) == i]).mean() for i in
                                    range(n_class)])
                else:
                    mus = np.array(
                        [(tii_hat[y_train_noisy[:len(tii_hat)] == i]).mean() for i in range(n_class)])
                criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args,
                                                                          use = args["method"],
                                                                          debug = False,
                                                                          weights = weights)


        elif args["method"] == "quantiles":
            print("Computing new quantiles and updating criterion")
            quantiles = get_noise_quantiles(tii_hat, y_train_noisy[:len(tii_hat)], args,
                                            dummy = dummy_labels)
            print("New quantiles :")
            print(quantiles)
            criterion = models_pytorch.loss_general_forward_iterative(S, quantiles, mus, args,
                                                                      use = args["method"],
                                                                      debug = False, weights = weights)

        # Test
        test_acc = evaluate(test_loader, model)

        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images: Model %.4f %% ' % (
                epoch + 1, args["nb_epoch"], len_test, test_acc))
        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ': ' + str(train_loss) + ' ' + str(train_acc) + ' ' + str(
                    test_acc) + "\n")

        if len(args["model_export"]) > 0:
            name = f"model_epoch{epoch}_acc{round(test_acc, 3)}_updmu" \
                   f"{str(args['update_mus'])}_truncbeta{str(args['truncate_ratios'])}_warmup" \
                   f"{warm_start}"
            path = os.path.join(args["model_export"], name)
            print("Saving trained noisy model at path: ", path)
            torch.save(model, path)


if __name__ == '__main__':
    main()
