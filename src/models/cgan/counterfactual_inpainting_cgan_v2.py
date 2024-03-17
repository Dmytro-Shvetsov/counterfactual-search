import torch
from torch.autograd import Variable

from src.losses import CARL, kl_divergence, loss_hinge_dis, loss_hinge_gen, tv_loss
from src.utils.grad_norm import grad_norm
from src.models.cgan.counterfactual_cgan import posterior2bin, CounterfactualCGAN

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class CounterfactualInpaintingCGANV2(CounterfactualCGAN):
    
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(img_size, opt, *args, **kwargs)
        self.lambda_tv = opt.get('lambda_tv', 0.0)
    
    def posterior_prob(self, x):
        f_x, f_x_discrete, _, _ = super().posterior_prob(x)
        f_x_desired = f_x.clone().detach()
        f_x_desired_discrete = f_x_discrete.clone().detach()
        
        # mask of what samples classifier predicted as `abnormal`
        inpaint_group = f_x_discrete.bool()
        # `abnormalities` need to be inpainted and classifier should predict `normal` on them
        f_x_desired[inpaint_group] = 1e-6
        f_x_desired_discrete[inpaint_group] = 0
        return f_x, f_x_discrete, f_x_desired, f_x_desired_discrete

    def reconstruction_loss(self, real_imgs, gen_imgs, masks, f_x_discrete, f_x_desired_discrete, z=None):
        forward_term = self.l1(real_imgs, gen_imgs)
        
        if not self.opt.get('cyclic_rec', False):
            return forward_term

        ifxc_fx = self.explanation_function(
            x=gen_imgs,  # I_f(x, c)
            f_x_discrete=f_x_desired_discrete, # f_x_desired_discrete is always zeros
        )
        # cyclic rec 1
        # L_rec(x, I_f(I_f(x, c), f(x)))
        # cyclic_term = self.l1(real_imgs, ifxc_fx)

        # cyclic rec 2
        cyclic_term = self.l1(gen_imgs, ifxc_fx)
        return forward_term + cyclic_term

    def forward(self, batch, training=False, validation=False, compute_norms=False, global_step=None):
        assert training and not validation or validation and not training

        # `real_imgs` and `gen_imgs` are in [-1, 1] range
        imgs, labels, masks = batch['image'], batch['label'], batch['masks']
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        masks = Variable(masks.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Classifier predictions and desired outputs for the explanation function
        with torch.no_grad():
            # technically condition `c` (real_f_x_desired) is now classifier driven choice to:
            # 1) `inpaint`  (real_f_x_discrete == 1)
            # 2) `identity` (real_f_x_discrete == 0)
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.posterior_prob(real_imgs)

        # -----------------
        #  Train Generator
        # -----------------
        if training:
            self.optimizer_G.zero_grad()

        # E(x)
        z = self.enc(real_imgs)
        # G(z, c) = I_f(x, f(x))
        gen_imgs = self.gen(z, real_f_x_discrete, x=real_imgs if self.ptb_based else None)

        update_generator = global_step is not None and global_step % self.gen_update_freq == 0
        
        if update_generator or validation:
            # data consistency loss for generator
            # discriminator is guided by the flipped prediction of the classifier on generated images
            # TODO: try passing f(x)
            # TODO: try passing f(x_cf)
            dis_fake = self.disc(gen_imgs, real_f_x_desired_discrete)
            if self.adv_loss == 'hinge':
                g_adv_loss = self.lambda_adv * loss_hinge_gen(dis_fake)
            else:
                g_adv_loss = self.lambda_adv * self.adversarial_loss(dis_fake, valid)

            # classifier consistency loss for generator
            # f(I_f(x, c)) ≈ c
            gen_f_x, _, _, _ = self.posterior_prob(gen_imgs)
            # both y_pred and y_target are single-value probs for class k
            g_kl = (
                self.lambda_kl * kl_divergence(gen_f_x, real_f_x_desired)
                if self.lambda_kl != 0 else torch.tensor(0.0, requires_grad=True)
            )
            # reconstruction loss for generator
            g_rec_loss = (
                self.lambda_rec * self.reconstruction_loss(real_imgs, gen_imgs, masks, real_f_x_discrete, real_f_x_desired_discrete, z=z)
                if self.lambda_rec != 0 else torch.tensor(0.0, requires_grad=True)
            )

            if self.lambda_tv != 0:
                g_tv = self.lambda_tv * tv_loss(torch.abs(real_imgs.add(1).div(2) - gen_imgs.add(1).div(2)).mul(255))
            else:
                g_tv = torch.tensor(0.0, requires_grad=True)
            # total generator loss
            g_loss = g_adv_loss + g_kl + g_rec_loss + g_tv

            # update generator
            if update_generator:
                self.fabric.backward(g_loss)
                if compute_norms:
                    self.norms['E'] = grad_norm(self.enc)
                    self.norms['G'] = grad_norm(self.gen)
                self.optimizer_G.step()

            self.gen_loss_logs['g_adv'] = g_adv_loss.item()
            self.gen_loss_logs['g_kl'] = g_kl.item()
            self.gen_loss_logs['g_rec_loss'] = g_rec_loss.item()
            self.gen_loss_logs['g_tv'] = g_tv.item()
            self.gen_loss_logs['g_loss'] = g_loss.item()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if training:
            self.optimizer_D.zero_grad()

        dis_real = self.disc(real_imgs, real_f_x_discrete) # changed from real_f_x_desired_discrete to real_f_x_discrete
        dis_fake = self.disc(gen_imgs.detach(), real_f_x_desired_discrete)

        # data consistency loss for discriminator (real and fake images)
        if self.adv_loss == 'hinge':
            d_real_loss, d_fake_loss = loss_hinge_dis(dis_fake, dis_real)
        else:
            d_real_loss = self.adversarial_loss(dis_real, valid)
            d_fake_loss = self.adversarial_loss(dis_fake, fake)
        
        # total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        if training:
            self.fabric.backward(d_loss)
            if compute_norms:
                self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()

        self.disc_loss_logs['d_real_loss'] = d_real_loss.item()
        self.disc_loss_logs['d_fake_loss'] = d_fake_loss.item()
        self.disc_loss_logs['d_loss'] = d_loss.item()

        outs = {
            'loss': {**self.gen_loss_logs, **self.disc_loss_logs},
            'gen_imgs': gen_imgs,
        }
        return outs
