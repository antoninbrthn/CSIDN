'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Loss functions, architectures, noisy model training.
'''

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from utils.misc_utils import *
import utils.noise_utils as noise_utils
from utils.calibration import ECE
import os
from torch.autograd import Variable


##############
## LOSSES
##############
def L1loss(dummy = False):
    def loss(y_pred, y_true):
        if dummy:
            return F.l1_loss(y_pred, y_true)
        else:
            y_true_oh = torch.FloatTensor(y_pred.size(0), y_pred.size(1))
            y_true_oh.zero_()
            y_true_oh.scatter_(1, y_true.cpu().view(-1, 1), 1)
            y_true_oh = y_true_oh.to("cuda:0")
            return F.l1_loss(y_pred, y_true_oh)

    return loss


def loss_forward(T):
    T = torch.tensor(T, dtype = torch.float32).to("cuda:0")

    def loss(y_pred, y_true):
        return F.cross_entropy(torch.matmul(y_pred, torch.t(T)), y_true)

    return loss


def loss_Lq(q, dummy = False):
    def loss(y_pred, y_true):
        if dummy:
            losses = torch.sum((1 - torch.pow((y_pred * y_true).sum(dim = 1), q)) / q)
        else:
            y_true_oh = torch.FloatTensor(y_pred.size(0), y_pred.size(1))
            y_true_oh.zero_()
            y_true_oh.scatter_(1, y_true.cpu().view(-1, 1), 1)
            y_true_oh = y_true_oh.to("cuda:0")
            losses = torch.sum((1 - torch.pow((y_pred * y_true_oh).sum(dim = 1), q)) / q)
        return losses

    return loss


def loss_DMI(dummy = False):
    def loss(y_pred, y_true):
        if dummy:
            mat = y_true @ y_pred
        else:
            y_true_oh = torch.FloatTensor(y_pred.size(0), y_pred.size(1))
            y_true_oh.zero_()
            y_true_oh.scatter_(1, y_true.cpu().view(-1, 1), 1)
            y_true_oh = y_true_oh.to("cuda:0")
            mat = y_true_oh.T @ y_pred
            assert mat.size(0) == 14
        losses = -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
        return losses

    return loss


def loss_Lq_cpu(q, dummy = False):
    def loss(y_pred, y_true):
        if dummy:
            losses = torch.sum((1 - torch.pow((y_pred * y_true).sum(dim = 1), q)) / q)
        else:
            y_true_oh = torch.FloatTensor(y_pred.size(0), y_pred.size(1))
            y_true_oh.zero_()
            y_true_oh.scatter_(1, y_true.view(-1, 1), 1)
            losses = torch.sum((1 - torch.pow((y_pred * y_true_oh).sum(dim = 1), q)) / q)
        return losses

    return loss


def loss_corrected_forward(S, args, bs = 32):
    def loss(y_pred, y_true, r):
        losses = 0
        for i in range(args["bs"]):
            P = noise_utils.S_to_T(S, eta_x = r.numpy()[i], n_class = args["n_class"])
            P = torch.tensor(P, dtype = torch.float32, requires_grad = False).to(args["device"])
            losses += torch.sum(y_true[i] * torch.log(torch.matmul(y_pred[i], torch.t(P))))
        return -losses

    return loss


def loss_general_forward(S, quantiles, args, bs = 32, use = "mean", debug = False):
    """
    if use = mean : take the 5th quantile every time
    if use = quantiles : take the quantile corresponding to the quantile of the assigned label
    """

    def loss(y_pred, y_true, r):
        losses = 0
        for i in range(args["bs"]):
            if use == "mean":
                diag = quantiles[:, 5]
            elif use == "quantiles":
                diag = get_diag_quantiles(r[i], y_true[i], quantiles)
            diag = torch.tensor(diag, dtype = torch.float32)
            mus_hat = get_mus_hat(r[i], y_true[i], diag)
            P = noise_utils.S_to_T_torch(torch.tensor(S, dtype = torch.float32), mus_hat,
                                         n_class = args["n_class"]).to(args["device"])
            losses += torch.sum(y_true[i] * torch.log(torch.matmul(y_pred[i], torch.t(P))))
        return -losses

    return loss


def loss_general_forward_iterative(S, quantiles, mus, args, use = "mean", debug = False, weights = None,
                                   dummy = False):
    """
    if use = mean : take mean from the mus input
    if use = quantiles : take the quantile corresponding to the quantile of the assigned label
    """
    max_conf = 0.99
    print("LOSS USING {}".format(use))
    print("mus", mus)

    def loss(y_pred, y_true, r, beta):
        losses = 0
        for i in range(args["bs"]):
            if use == "mean":
                diag = mus
            elif use == "median":
                diag = quantiles[:, 5]
            elif use == "quantiles":
                diag = get_diag_quantiles(r[i], y_true[i], quantiles)
            else:
                raise Exception

            diag = torch.tensor(diag, dtype = torch.float32)
            conf = torch.tensor(min(max_conf, r[i] * beta[i])).to(args["device"])
            mus_hat = get_mus_hat(conf, y_true[i], diag)
            if debug:
                p = True
            else:
                p = False
            P = noise_utils.S_to_T_torch(torch.tensor(S, dtype = torch.float32), mus_hat,
                                         n_class = args["n_class"], plot = p).to(args["device"])
            if dummy == False:
                y_true_i = torch.zeros(args["n_class"])
                y_true_i[y_true[i]] = 1
                ind = y_true[i].item()
            else:
                y_true_i = y_true[i]
                ind = y_true[i].argmax().item()
            y_true_i = y_true_i.to(args["device"])
            # Weighted loss
            if weights is not None:
                losses += weights[ind] * torch.sum(
                    y_true_i * torch.log(torch.matmul(torch.t(P), y_pred[i])))
            else:
                losses += torch.sum(y_true_i * torch.log(torch.matmul(torch.t(P), y_pred[i])))
        if np.random.rand() < -0.005:
            if use == "median":
                diag_opposite = mus
            elif use == "mean":
                diag_opposite = quantiles[:, 5]
            else:
                raise Exception
            diag_opposite = torch.tensor(diag_opposite, dtype = torch.float32)
            mus_hat_opposite = get_mus_hat(conf, y_true[i], diag_opposite)
            P_opo = noise_utils.S_to_T_torch(torch.tensor(S, dtype = torch.float32), mus_hat_opposite,
                                             n_class = args["n_class"], plot = p).to(args["device"])

            # Compare losses
            l = weights[ind] * torch.sum(y_true_i * torch.log(torch.matmul(y_pred[i], torch.t(P))))
            l_opo = weights[ind] * torch.sum(
                y_true_i * torch.log(torch.matmul(y_pred[i], torch.t(P_opo))))

            print("Currently using diag : ", diag)
            print("OPPOSITE diag : ", diag_opposite)
            print("Currently using mus_hat : ", mus_hat)
            print("OPPOSITE mus_hat : ", mus_hat_opposite)
            print("Current P - Opposite P: ", P - P_opo)
            print("Y_true_i : ", y_true_i)
            print("Current log part:", torch.log(torch.matmul(y_pred[i], torch.t(P))))
            print("OPPOSITE log part:", torch.log(torch.matmul(y_pred[i], torch.t(P_opo))))
            print("Current presum part:", y_true_i * torch.log(torch.matmul(y_pred[i], torch.t(P))))
            print("OPPOSITE presum part:",
                  y_true_i * torch.log(torch.matmul(y_pred[i], torch.t(P_opo))))
            print("Current loss:", l)
            print("OPPOSITE loss:", l_opo)
        return -losses

    return loss


def loss_general_forward_iterative_wLQ(S, quantiles, mus, args, use = "mean", debug = False,
                                       weights = None,
                                       dummy = False):
    """
    if use = mean : take mean from the mus input
    if use = quantiles : take the quantile corresponding to the quantile of the assigned label
    """
    q = 0.7
    max_conf = torch.tensor(0.99)
    print("LOSS USING {} + LQ norm".format(use))
    print("mus", mus)

    def loss(y_pred, y_true, r, beta):
        losses = 0
        for i in range(args["bs"]):
            if use == "mean":
                diag = mus
            elif use == "median":
                diag = quantiles[:, 5]
            elif use == "quantiles":
                diag = get_diag_quantiles(r[i], y_true[i], quantiles)
            else:
                raise Exception

            diag = torch.tensor(diag, dtype = torch.float32)
            conf = min(max_conf, r[i] * beta[i]).cpu()
            mus_hat = get_mus_hat(conf, y_true[i], diag)
            if debug:
                p = True
            else:
                p = False
            P = noise_utils.S_to_T_torch(torch.tensor(S, dtype = torch.float32), mus_hat,
                                         n_class = args["n_class"], plot = p).to(args["device"])
            if dummy == False:
                y_true_i = torch.zeros(args["n_class"])
                y_true_i[y_true[i]] = 1
                ind = y_true[i].item()
            else:
                y_true_i = y_true[i]
                ind = y_true[i].argmax().item()
            y_true_i = y_true_i.to(args["device"])
            y_pred_corr_i = torch.matmul(torch.t(P), y_pred[i])

            # Weighted loss
            if weights is not None:
                losses += weights[ind] * (1 - torch.pow((y_pred_corr_i * y_true_i).sum(
                    dim = 0), q)) / q
            else:
                losses += (1 - torch.pow((y_pred_corr_i * y_true_i).sum(dim = 0), q)) / q

        return losses

    return loss


##############
## MODELS
##############
class base_model(nn.Module):
    def __init__(self, args):
        super(base_model, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, args["n_class"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = 1)
        return x


class base_model_mnist(nn.Module):
    def __init__(self, args):
        super(base_model_mnist, self).__init__()
        n_features = args["n_features"]
        self.fc1 = nn.Linear(args["input_size"], n_features)
        self.fc1_bn = nn.BatchNorm1d(n_features)
        self.fc2 = nn.Linear(n_features, n_features)
        self.fc2_bn = nn.BatchNorm1d(n_features)
        self.fc3 = nn.Linear(n_features, args["n_class"])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


def call_bn(bn, x):
    return bn(x)


class base_cnn_mnist(nn.Module):
    def __init__(self, args, dropout_rate = 0.0, top_bn = False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        self.size = int(args["input_size"] ** 0.5)
        super(base_cnn_mnist, self).__init__()
        self.c1 = nn.Conv2d(args["input_channel"], args["n_features"], kernel_size = 3, stride = 1,
                            padding = 1)
        self.c2 = nn.Conv2d(args["n_features"], args["n_features"], kernel_size = 3, stride = 1,
                            padding = 1)
        self.l_c1 = nn.Linear(args["n_features"], args["n_class"])
        self.bn1 = nn.BatchNorm2d(args["n_features"])
        self.bn2 = nn.BatchNorm2d(args["n_features"])
        self.T_star = 1

    def forward_base(self, x):
        # x = x.view(-1, , self.size, self.size)
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope = 0.01)
        h = F.max_pool2d(h, kernel_size = 2, stride = 2)
        h = F.dropout2d(h, p = self.dropout_rate)

        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope = 0.01)
        h = F.max_pool2d(h, kernel_size = 2, stride = 2)
        h = F.dropout2d(h, p = self.dropout_rate)

        h = F.avg_pool2d(h, kernel_size = h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        return logit

    def forward(self, x):
        return F.softmax(self.forward_base(x), dim = 1)

    def forward_TS(self, x):
        return F.softmax(self.forward_base(x) / self.T_star, dim = 1)

    def forward_TS_np(self, x, T):
        return F.softmax(self.forward_base(x) / T[0], dim = 1)

    def forward_TS_custom(self, x, T):
        return F.softmax(self.forward_base(x) / T, dim = 1)


class large_CNN(nn.Module):
    def __init__(self, args, dropout_rate = 0.25, top_bn = False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(large_CNN, self).__init__()
        self.c1 = nn.Conv2d(args["input_channel"], args["n_features"], kernel_size = 3, stride = 1,
                            padding = 1)
        self.c2 = nn.Conv2d(args["n_features"], args["n_features"], kernel_size = 3, stride = 1,
                            padding = 1)
        self.c3 = nn.Conv2d(args["n_features"], args["n_features"], kernel_size = 3, stride = 1,
                            padding = 1)
        self.c4 = nn.Conv2d(args["n_features"], args["n_features"] * 2, kernel_size = 3, stride = 1,
                            padding = 1)
        self.c5 = nn.Conv2d(args["n_features"] * 2, args["n_features"] * 2, kernel_size = 3, stride = 1,
                            padding = 1)
        self.c6 = nn.Conv2d(args["n_features"] * 2, args["n_features"] * 2, kernel_size = 3, stride = 1,
                            padding = 1)
        self.c7 = nn.Conv2d(args["n_features"] * 2, args["n_features"] * 4, kernel_size = 3, stride = 1,
                            padding = 0)
        self.c8 = nn.Conv2d(args["n_features"] * 4, args["n_features"] * 2, kernel_size = 3, stride = 1,
                            padding = 0)
        self.c9 = nn.Conv2d(args["n_features"] * 2, args["n_features"], kernel_size = 3, stride = 1,
                            padding = 0)
        self.l_c1 = nn.Linear(args["n_features"], args["n_class"])
        self.bn1 = nn.BatchNorm2d(args["n_features"])
        self.bn2 = nn.BatchNorm2d(args["n_features"])
        self.bn3 = nn.BatchNorm2d(args["n_features"])
        self.bn4 = nn.BatchNorm2d(args["n_features"] * 2)
        self.bn5 = nn.BatchNorm2d(args["n_features"] * 2)
        self.bn6 = nn.BatchNorm2d(args["n_features"] * 2)
        self.bn7 = nn.BatchNorm2d(args["n_features"] * 4)
        self.bn8 = nn.BatchNorm2d(args["n_features"] * 2)
        self.bn9 = nn.BatchNorm2d(args["n_features"])

        self.T_star = 1

    def forward_base(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope = 0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope = 0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope = 0.01)
        h = F.max_pool2d(h, kernel_size = 2, stride = 2)
        h = F.dropout2d(h, p = self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope = 0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope = 0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope = 0.01)
        h = F.max_pool2d(h, kernel_size = 2, stride = 2)
        h = F.dropout2d(h, p = self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope = 0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope = 0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope = 0.01)
        h = F.avg_pool2d(h, kernel_size = h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit

    def forward(self, x):
        return F.softmax(self.forward_base(x), dim = 1)

    def forward_TS(self, x):
        return F.softmax(self.forward_base(x) / self.T_star, dim = 1)

    def forward_TS_custom(self, x, T):
        return F.softmax(self.forward_base(x) / T, dim = 1)


class resnet18(nn.Module):
    def __init__(self, args, pretrained = True):
        super(resnet18, self).__init__()
        print(f"Importing {'pretrained' * pretrained}{'vanilla' * ~pretrained} ResNet18")
        self.model = models.resnet18(pretrained = pretrained)
        self.model.fc = nn.Linear(512, args["n_class"])

        self.T_star = 1

    def forward_base(self, x):
        logit = self.model(x)
        return logit

    def forward(self, x):
        return F.softmax(self.forward_base(x), dim = 1)

    def forward_TS(self, x):
        return F.softmax(self.forward_base(x) / self.T_star, dim = 1)

    def forward_TS_np(self, x, T):
        return F.softmax(self.forward_base(x) / T[0], dim = 1)

    def forward_TS_custom(self, x, T):
        return F.softmax(self.forward_base(x) / T, dim = 1)


class resnet18_temp(nn.Module):
    def __init__(self, args, pretrained = True, T = 1):
        super(resnet18_temp, self).__init__()
        print(f"Importing {'pretrained' * pretrained}{'vanilla' * ~pretrained} ResNet18 with Temp Scaling")
        self.model = models.resnet18(pretrained = pretrained)
        self.model.fc = nn.Linear(512, args["n_class"])

        self.T_star = T
        print(f'Will use temperature scaling. Current T={self.T_star}')

    def forward_base(self, x):
        logit = self.model(x)
        return logit

    def forward(self, x):
        return F.softmax(self.forward_base(x) / self.T_star, dim = 1)

    def forward_TS_custom(self, x, T):
        return F.softmax(self.forward_base(x) / T, dim = 1)


class resnet50(nn.Module):
    def __init__(self, args, pretrained = True):
        super(resnet50, self).__init__()
        print(f"Importing {'pretrained' * pretrained}{'vanilla' * ~pretrained} ResNet50")
        self.model = models.resnet50(pretrained = pretrained)
        self.model.fc = nn.Linear(2048, args["n_class"])
        self.T_star = 1

    def forward_base(self, x):
        logit = self.model(x)
        return logit

    def forward(self, x):
        return F.softmax(self.forward_base(x), dim = 1)

    def forward_TS(self, x):
        return F.softmax(self.forward_base(x) / self.T_star, dim = 1)

    def forward_TS_np(self, x, T):
        return F.softmax(self.forward_base(x) / T[0], dim = 1)

    def forward_TS_custom(self, x, T):
        return F.softmax(self.forward_base(x) / T, dim = 1)


###############
## TOOLS
###############
def find_quantiles(r, quantiles):
    for i, q in enumerate(quantiles[1:]):
        if r < q:
            return i
    return len(quantiles) - 1


def get_diag_quantiles(r, y, quantiles):
    curr_quantile = find_quantiles(r, quantiles[y.argmax()])
    return quantiles[:, curr_quantile]


def get_mus_hat(r, y, diag_hat):
    if len(y.size()) == 0:
        y_vec = torch.zeros_like(diag_hat)
        y_vec[y] = 1
    else:
        y_vec = y
    try:
        r = r.cpu()
    except:
        pass
    return (1 - y_vec) * diag_hat + r * y_vec


####################
## Naive classifier
####################
def train_naive_nn(base_model, trainloader, testloader, args, nb_epoch = 10, lr = 0.01,
                   x_test = None, y_test = None, device = "cpu",
                   weights = None, show_ece = False, clothing_exp = False, train_limit = None,
                   import_path = '', export_path = '', T = None):
    if train_limit is None:
        train_limit = len(trainloader)
    naive_h = base_model(args)
    naive_h.to(device)

    if len(import_path) > 0:
        try:
            naive_h = torch.load(import_path)
            print("Loaded noisy model from: ", import_path)
            print(f"Training it on {nb_epoch}")
        except Exception as err:
            print("Unable to load noisy model checkpoint. Error:", err)

    criterion = F.cross_entropy
    optimizer = optim.Adam(naive_h.parameters(), lr = lr)

    # Training
    for epoch in range(nb_epoch):
        current_loss = 0.0

        for i, data in enumerate(trainloader):
            if i > train_limit:
                break
            if clothing_exp:
                images, labels_noisy, conf = data
            else:
                images, labels, labels_noisy, _ = data

            images = Variable(images).to(device)
            labels_noisy = Variable(labels_noisy).to(device).long()

            optimizer.zero_grad()

            output = naive_h(images)

            if args["dummy_labels"] == False:
                loss = criterion(output, labels_noisy, weight = weights)
            else:
                loss = criterion(output, to_int(labels_noisy), weight = weights)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
        print('[%d, %5d/%5d] loss: %.3f' %
              (epoch + 1, i + 1, len(trainloader), current_loss))

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                if clothing_exp:
                    inputs, label, _ = data
                else:
                    inputs, label = data

                inputs = Variable(inputs).to(device)
                label = Variable(label).to(device)

                output = naive_h(inputs)
                predicted = torch.argmax(output.data, dim = 1)
                total += label.size(0)
                if args["dummy_labels"]:
                    true = torch.argmax(label.data, dim = 1)
                else:
                    true = label.data
                correct += (predicted == true).sum().item()
        print('Accuracy of the naive network after {} epochs : {}%'.format(epoch,
                                                                           100 * correct / total))
        if show_ece:
            logits, y_true = get_logits(naive_h, testloader, T = 1, max_batch = 50, device = device)
            logits_t, y_true_t = torch.tensor(logits), torch.tensor(y_true)
            print("Naive noisy classifier ECE: ", ECE(logits_t, y_true_t, nb_bins = 20))

        if len(export_path) > 0:
            name = f"noisymodel_epoch{epoch}_acc{round(100 * correct / total, 2)}"
            path = os.path.join(export_path, name)
            print("Saving trained noisy model at path: ", path)
            torch.save(naive_h, path)

    plot_interpr = False
    if plot_interpr:
        intervals, acc = get_reliability_diag(naive_h(torch.tensor(x_test)),
                                              torch.tensor(y_test).argmax(dim = 1), nb_bins = 10)
        plt.bar(intervals, acc, width = 0.1)
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        plt.title('Naive classifier - Epoch {}'.format(epoch))
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.xlim([0, 1])
        plt.show()

    return naive_h


def get_logits(model, loader, T = 1, max_batch = 100, device = 'cpu'):
    all_logit = []
    y_pred = []
    y_true = []
    for i, data in enumerate(loader):
        img, lab, _ = data
        logit = model.forward_TS_custom(x = img.to(device), T = T)
        logit_np = logit.cpu().detach().numpy()
        if len(all_logit) == 0:
            all_logit = logit_np
        else:
            all_logit = np.concatenate((all_logit, logit_np))
        y_true += list(lab.numpy())

        del img, lab, logit
        if i >= max_batch:
            break

    y_true = np.array(y_true)
    return all_logit, y_true
