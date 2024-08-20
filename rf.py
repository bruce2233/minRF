# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import ml_collections
from dataset import get_ds
from tqdm import tqdm
 
config = ml_collections.ConfigDict()
config.data = data = ml_collections.ConfigDict()
data.image_size = 32
# data.image_size = 224
data.num_channels = 3
# data.dataset = 'sketchy'
data.dataset = 'sketchy32'
# data.dataset = 'cifar'

config.training = training = ml_collections.ConfigDict()
training.batch_size = 32

ds, transform, model = get_ds(config, data.dataset)
# ds = fdatasets(transform=transform)
class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        if type(x) is list:
            z1 = x[0]
            x = x[1]
        else:
            z1 = torch.randn_like(x)
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama
    from glob import glob
    import os, logging
    
    logging.basicConfig()
    
    results_dir = "results"
    z1_type = "z1"
    # z1_type = "noise"
    
    parser = argparse.ArgumentParser(description="use cifar?")
    experiment_index = len(glob(f"{results_dir}/*"))
    experiment_dir = f"{results_dir}/{experiment_index:03d}-{config.data.dataset.replace('/','-')}"
    os.makedirs(experiment_dir, exist_ok=True)
    logging.info(f"experiment dir: {experiment_dir}")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    # mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(ds, batch_size=config.training.batch_size, shuffle=True, drop_last=True)

    wandb.init(project=f"rf_{config.data.dataset}")

    for epoch in tqdm(range(1000)):

        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            # if i == 10:
            #     break
            if type(x) is list:
                x = [i.cuda() for i in x]
            else: 
                x = x.cuda()
            c = c.cuda()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        #%%
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, config.training.batch_size).cuda() % 10
            uncond = torch.ones_like(cond) * 10

            if z1_type == 'noise':
                z1_eval = torch.randn(config.training.batch_size, config.data.num_channels, 32, 32).cuda()
            elif z1_type == 'z1':
                z1_eval = next(iter(dataloader))[0][0].cuda()
            images = rf.sample(z1_eval, cond, uncond)
            # image sequences to gif
            gif = []
            for image in images:
                # unnormalize
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"{experiment_dir}/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"{experiment_dir}/sample_{epoch}_last.png")

        rf.model.train()
