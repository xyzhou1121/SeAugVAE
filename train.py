from __future__ import print_function
import datetime
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from network import SeAugVAE, weights_init, ImageDiscriminator, FeatureDiscriminator
from datasetloader import trainImageDataset, testImageDataset

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformation
data_transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.Resize(224),

    ]),
    "test": transforms.Compose([
        transforms.Resize(224),
    ])}

# Dataset root
image_path = r"/media/admin1/xyzhou/Response/DataSet/"

# Batch size
batch_size = 8

# Train dataset
train_dataset = trainImageDataset(root=image_path + "train",
                                  transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

# SeAugVAE model
modelV = SeAugVAE().to(device)
modelI = ImageDiscriminator().to(device)
modelF = FeatureDiscriminator().to(device)
modelV.apply(weights_init)
modelI.apply(weights_init)
modelF.apply(weights_init)
optimizerV = optim.Adam(modelV.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerI = optim.Adam(modelI.parameters(), lr=4e-4, betas=(0.5, 0.999))
optimizerF = optim.Adam(modelF.parameters(), lr=2e-4, betas=(0.5, 0.999))


# Loss
l_bce = nn.BCEWithLogitsLoss()

# Loss weights
w_adv_i = 1
w_con = 1
w_kl = 1
w_kl_aug = 0.1
w_adv_f = 1

w_d = 0.5
w_f = 0.5


# parameter optimiztion
def forward_V(input, lam):
    """ Forward propagate through netE and netD
    """
    fake, mu, logvar, z, _ = modelV(input)

    disturb = torch.distributions.normal.Normal(z, lam)
    noise = disturb.sample().to(device=z.device)
    fake_noise = noise.detach()
    fake_aug = modelV.decode(fake_noise)

    fake_input = fake_aug.detach()
    mu_aug, logvar_aug, _, _ = modelV.encode(fake_input)
    fake_z = modelV.reparameterize(mu_aug, logvar_aug)

    return fake, mu, logvar, z, fake_z, mu_aug, logvar_aug, fake_aug


##
def forward_I(input, fake):
    """ Forward propagate through netI
    """
    pred_real, feat_real = modelI(input)
    pred_fake, feat_fake = modelI(fake.detach())
    return pred_real, feat_real, pred_fake, feat_fake


def forward_F(z, fake_z):
    """ Forward propagate through netF
    """
    z_pred_real = modelF(z.detach())
    z_pred_fake = modelF(fake_z.detach())
    return z_pred_real, z_pred_fake


##
def backward_V(input, fake, mu, logvar, fake_z, mu_aug, logvar_aug, fake_aug):
    """ Backpropagate through netE and netD
    """

    real_label_ = torch.ones(size=(fake_z.shape[0], 1), dtype=torch.float32, device=device)

    err_v_con = torch.mean(torch.sum(torch.abs(input - fake), dim=(1, 2, 3)))
    err_v_kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    err_v_aug_kl = torch.mean(-0.5 * torch.sum(1 + logvar_aug - mu_aug ** 2 - logvar_aug.exp(), dim=1), dim=0)
    err_adv_i = l_bce(modelI(fake_aug)[0], real_label_)
    err_adv_f = l_bce(modelF(fake_z), real_label_)
    err_v = err_adv_i * w_adv_i + err_v_con * w_con + err_v_kl * w_kl + err_adv_f * w_adv_f + err_v_aug_kl * w_kl_aug
    err_v.backward()
    return err_v, err_v_con


##
def backward_I(pred_real, pred_fake):
    """ Backpropagate through netI
    """
    real_label = torch.ones(size=(pred_real.shape[0], 1), dtype=torch.float32, device=device)
    fake_label = torch.zeros(size=(pred_fake.shape[0], 1), dtype=torch.float32, device=device)

    # Real - Fake Loss
    err_d_real = l_bce(pred_real, real_label)
    err_d_fake = l_bce(pred_fake, fake_label)

    # NetD Loss & Backward-Pass
    err_d = (err_d_real + err_d_fake) * w_d
    err_d.backward()
    return err_d


def backward_F(z_pred_real, z_pred_fake):
    """ Backpropagate through netF
    """
    z_real_label = torch.ones(size=(z_pred_real.shape[0], 1), dtype=torch.float32, device=device)
    z_fake_label = torch.zeros(size=(z_pred_fake.shape[0], 1), dtype=torch.float32, device=device)

    # Real - Fake Loss
    err_f_real = l_bce(z_pred_real, z_real_label)
    err_f_fake = l_bce(z_pred_fake, z_fake_label)

    # NetD Loss & Backward-Pass
    err_f = (err_f_real + err_f_fake) * w_f
    err_f.backward()
    return err_f


##
def reinit_I():
    """ Re-initialize the weights of netI
    """
    modelI.apply(weights_init)
    print('   Reloading net d')


def reinit_F():
    """ Re-initialize the weights of netF
    """
    modelF.apply(weights_init)
    print('   Reloading net f')


def optimize_params(input, lam):
    """ Forwardpass, Loss Computation and Backwardpass.
    """
    # Forward-pass
    fake, mu, logvar, z, fake_z, mu_aug, logvar_aug, fake_aug = forward_V(input, lam)
    pred_real, feat_real, pred_fake, feat_fake = forward_I(input, fake_aug)
    z_pred_real, z_pred_fake = forward_F(z, fake_z)

    # Backward-pass
    # netE and netD
    optimizerV.zero_grad()
    err_v, err_v_enc = backward_V(input, fake, mu, logvar, fake_z, mu_aug, logvar_aug, fake_aug)
    optimizerV.step()

    # netI
    optimizerI.zero_grad()
    err_d = backward_I(pred_real, pred_fake)
    optimizerI.step()

    # netF
    optimizerF.zero_grad()
    err_f = backward_F(z_pred_real, z_pred_fake)
    optimizerF.step()

    if err_d.item() < 1e-5:
        reinit_I()

    if err_f.item() < 1e-5:
        reinit_F()

    return err_v


def train(epoch):
    """ One epoch of model training. """
    modelV.train()
    modelI.train()
    modelF.train()

    train_v_loss = 0

    # 以batch为单位训练
    for batch_idx, data in enumerate(train_loader):

        data = data['image']
        data = data.to(device)

        lam = 1
        err_v = optimize_params(data, lam)

        train_v_loss += err_v.item()

        rate = (batch_idx + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, err_v.item()), end="")

    print('====> Epoch: {} Average V_loss: {:.6f}'.format(
        int(epoch + 1), train_v_loss / len(train_loader.dataset)), end="")


if __name__ == '__main__':

    for epoch in range(0, 1000):
        start_time = datetime.datetime.now()
        train(epoch)
        end_time = datetime.datetime.now()
        time = str(end_time - start_time)
        print("time：" + time[0:7])
        print("-" * 10)
        print(' ')
        model_name = "SeAugVAE"
        if (epoch + 1) == 1000:
            # save net_state
            save_paths = r'/media/admin1/xyzhou/Response/Ours/pths/{}.pth'.format(
                model_name + "_NetV" + str((epoch + 1)))
            torch.save(modelV.state_dict(), save_paths)
            save_pathd = r'/media/admin1/xyzhou/Response/Ours/pths/{}.pth'.format(
                model_name + "_NetI" + str((epoch + 1)))
            torch.save(modelI.state_dict(), save_pathd)
            save_pathf = r'/media/admin1/xyzhou/Response/Ours/pths/{}.pth'.format(
                model_name + "_NetF" + str((epoch + 1)))
            torch.save(modelF.state_dict(), save_pathf)
