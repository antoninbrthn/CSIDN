'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Generation and export of synthetic dataset.
'''

import seaborn as sns

import utils.data_utils as data_utils
import utils.noise_utils as noise_utils
from utils.misc_utils import *
import matplotlib as mpl
import os

mpl.rcParams['scatter.marker'] = "o"
mpl.rcParams.update({'font.size': 22})
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['lines.markersize'] = 10

cmap = mpl.cm.get_cmap('Spectral')
custom_colors = [cmap(0.25), cmap(0.999), cmap(0.001)]
n_lim = 3000

n = 21000
val_split = 1 - 20 / 21
n_class = 3

# Mean corruption vectors
mus_tab = np.arange(0.1, 0.55, 0.05)

noise_type = "flip"
lr = 1e-2
q = 0.7
args = {"n_class": n_class,
        "noise_type": noise_type,
        "lr": lr}

# Generate data
data = data_utils.circle_data(n, val_split, n_class)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# Apply noise
# Flip
if args["noise_type"] == "flip":
    S = 0.1 * 1 * np.array([[0, 10, 0],
                            [0, 0, 10],
                            [10, 0, 0]])

# Symmetric
elif args["noise_type"] == "sym":
    S = 0.1 * 1 / 2 * np.array([[0, 10, 10],
                                [10, 0, 10],
                                [10, 10, 0]])

# Estimate of the mean corruption vector
mus_hat = np.zeros(n_class)


def assign_bucket(x, buckets):
    output = []
    for xi in x:
        i = 1
        for b in buckets:
            if xi < b:
                break
            else:
                i += 1
        output.append(i)
    return np.array(output)


def export_dataset(dt_name, x, y_true, y_noisy, conf):
    # Get stats
    acc = (y_true == y_noisy).mean()
    length = len(y_true)
    save_dir = "Synthetic/data/{}".format(dt_name)
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)

    with open("{}/infos.txt".format(save_dir), 'w+') as f:
        f.write("Number of instances : %d\nAccuracy : %.3f" % (length, acc))

    # Export
    with open("{}/x".format(save_dir), 'wb+') as f:
        np.save(f, x)

    with open("{}/y_noisy".format(save_dir), 'wb+') as f:
        np.save(f, y_noisy)

    with open("{}/y_true".format(save_dir), 'wb+') as f:
        np.save(f, y_true)

    with open("{}/r".format(save_dir), 'wb+') as f:
        np.save(f, conf)


for mus in mus_tab:
    print("-- Noise : {}".format(mus))
    T = noise_utils.S_to_T(S, mus = 1 - mus, n_class = n_class)
    print("Transition matrix: \n", T)

    ############
    ## Create noise
    ############

    ## Uniform noise
    y_noisy_uni = noise_utils.apply_class_noise(y_train, T, n_class)

    ## Non-uniform noise
    noise = noise_utils.directional_instance_noise(x_train, y_train, max_noise = 2 * mus)
    conf = 1 - noise
    print(noise.mean())
    y_noisy_nonuni = noise_utils.apply_noise(y_train, conf, S, n_class)
    r_train = conf

    # Get loaders
    train_loader, test_loader = data_utils.get_loader(x_train, y_train, y_noisy_nonuni, r_train, x_test, y_test)

    plot_noise = True
    if plot_noise:
        plt.figure(figsize = (7, 7))
        sns.scatterplot(x = x_train[:n_lim, 0], y = x_train[:n_lim, 1], hue = to_int(y_train)[:n_lim],
                        palette = custom_colors, alpha = 0.8)
        plt.title("Clean distribution")
        plt.legend(fontsize = 15)
        plt.show()
        plt.figure(figsize = (7, 7))
        sns.scatterplot(x = x_train[:n_lim, 0], y = x_train[:n_lim, 1], hue = to_int(y_noisy_uni)[:n_lim],
                        palette = custom_colors, alpha = 0.8)
        plt.title("Class-Conditional Noise")
        plt.legend(fontsize = 15)
        plt.show()
        plt.figure(figsize = (7, 7))
        sns.scatterplot(x = x_train[:n_lim, 0], y = x_train[:n_lim, 1], hue = to_int(y_noisy_nonuni)[:n_lim],
                        palette = custom_colors, alpha = 0.8)
        plt.title("Instance-Dependent Noise")
        plt.legend(fontsize = 15)
        plt.show()

        n_buckets = 10
        buckets = np.linspace(0, 1, n_buckets + 1)[1:]
        noises_buckets = assign_bucket(noise, buckets)
        f = lambda x: x ** 0.5
        opacity = lambda i: (f((n_buckets - (i + 1)) / n_buckets)) / (f((n_buckets - 1) / n_buckets))
        weight = [(noises_buckets == i).mean() for i in range(1, n_buckets)]
        # Variable opacity plot
        plt.figure(figsize = (7, 7))
        for i in range(1, n_buckets):
            y = to_int(y_noisy_nonuni[noises_buckets == i])
            x = x_train[noises_buckets == i]
            if i == 1:
                sns.scatterplot(x = x[:int(weight[i - 1] * n_lim), 0], y = x[:int(weight[i - 1] * n_lim), 1],
                                hue = y[:int(weight[i - 1] * n_lim)],
                                palette = custom_colors, alpha = opacity(i))
            else:
                sns.scatterplot(x = x[:int(weight[i - 1] * n_lim), 0], y = x[:int(weight[i - 1] * n_lim), 1],
                                hue = y[:int(weight[i - 1] * n_lim)], legend = False,
                                palette = custom_colors, alpha = opacity(i))
        plt.title("Instance-Dependent Noise + Confidence")
        plt.legend(fontsize = 15)
        plt.show()
    export = True
    if export:
        export_dataset("concentric_%s-%.2f" % (noise_type, mus), x_train, to_int(y_train), to_int(y_noisy_nonuni),
                       r_train)
