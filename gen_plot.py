import os
import json
import matplotlib.pyplot as plt
import numpy as np
clt_folder = "samples_test_robustness/Loss_clt/"
simple_folder = "samples_test_robustness/Loss_simple/"
color = ["red","blue"]
name = ["DVI Diffusion", "Simple Diffusion"]
def make_plot(data: list[float]):
    for i in range(len(data)):
        loss = data[i]
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
        mu = np.array(loss).mean(axis=0)
        sigma = np.array(loss).std(axis=0, ddof=1)
        stderr = sigma / np.sqrt(len(loss))
        axes.set_title("Epoch mean loss")
        axes.plot(
            range(len(loss[0])),
            mu,
            label=f" {color[i]} mean loss",
            color=color[i],
        )
        axes.fill_between(
            range(len(loss[0])),
            mu + stderr,
            mu - stderr,
            facecolor=color[i],
            alpha=0.3,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    fig.set_size_inches(16, 12)
    plt.savefig(f"mean_loss.png", dpi=100)




if __name__ == "__main__":
    clt_loss_files = os.listdir(clt_folder)
    clt_loss = []

    simple_loss_files = os.listdir("samples_test_robustness/Loss_simple/")
    simple_loss = []

    for filename in clt_loss_files:
        with open(os.path.join(clt_folder,filename), "r") as file:
            clt_loss.append(json.load(file))

    for filename in simple_loss_files:
        with open(os.path.join(simple_folder,filename), "r") as file:
            simple_loss.append(json.load(file))
    make_plot([simple_loss])


