# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from ast import arg
from email.policy import default
import os
import configargparse
import torch
import torch.nn as nn
from pathlib import Path
import wandb
from datasets.base import build_dataloader
from models.unet import UNet
from models.patch_gan_discriminator import NLayerDiscriminator
from models.video_discriminator import Discriminator_3D
from losses.loss import Loss
from losses.gan_loss import GANLoss
from utils.utils import create_image_pair
from metrics.metrics import Metrics
import shutil
import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def print_options(opt, parser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)


    file_name = os.path.join(opt.checkpoints_dir, 'GAN_config.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def config_parser():
    parser = configargparse.ArgumentParser()
    # config file
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    # training and model options
    parser.add_argument("--optimizer", type=str, help='choose optimizer type', default='adam')
    parser.add_argument("--num_epochs", type=int, default=61, help="number of epochs of training")
    parser.add_argument("--num_steps", type=int, default=25000, help="number of steps of training")
    parser.add_argument("--continue_from_epoch", type=int, default=0, help="Continue training from epoch (default=0)")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate") # 0.00001
    parser.add_argument("--lr_decay_steps", type=int, default=10, help="learning rate decay steps in epochs")
    parser.add_argument("--lr_disc", type=float, default=0.001, help="discriminator learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers to use during batch generation")
    parser.add_argument("--num_input_channels", type=int, default=3, help="number of input image channels")
    parser.add_argument("--num_output_channels", type=int, default=3, help="number of output image channels")
    parser.add_argument("--loss_names", nargs="*", type=str, default=['perceptual_loss_vgg'], help="names of losses used in training")
    parser.add_argument("--loss_weights", nargs="*", type=float, default=[1.0], help="weights assigned to losses in the order")
    parser.add_argument("--metric_names", nargs="*", type=str, default=['mean_absolute_error'], help="names of metrics to be logged")
    parser.add_argument("--metric_weights", nargs="*", type=float, default=[1.0], help="weights assigned to metrics in the order")
    parser.add_argument("--use_discriminator", action='store_true', help="choose if to use discriminator network")
    parser.add_argument("--use_video_discriminator", action='store_true', help="choose if to use video discriminator network")
    parser.add_argument("--steps_train_video_discr", type=int, default=1, help='frequency of video discriminator training')
    parser.add_argument("--loss_gan_video_weight", type=float, default=1, help='weight assigned to video discriminator loss')
    parser.add_argument("--use_label_maps", action='store_true', help="choose if to use label maps for discriminator")
    # dataset options
    parser.add_argument("--dataset_type", type=str, help="options: CustomDataset", default='CustomDataset') 
    parser.add_argument("--input_train_root_dir", type=str, help="Path to training input images", default='./data/input_images_train') 
    parser.add_argument("--output_train_root_dir", type=str, help="Path to training output images", default='./data/output_images_train')  
    parser.add_argument("--label_train_root_dir", type=str, help="Path to training label images", default='./data/label_images_train')  
    parser.add_argument("--input_val_root_dir", type=str, help="Path to val input images", default=None) 
    parser.add_argument("--output_val_root_dir", type=str, help="Path to val output images", default=None)
    parser.add_argument("--label_val_root_dir", type=str, help="Path to val label images", default=None)  
    parser.add_argument("--width", type=int, default=640, help="width")
    parser.add_argument("--height", type=int, default=360, help="height")
    # logging/saving options
    parser.add_argument("--checkpoints_dir", type=str, help='specify the directory to save the model', default='./chkpts/experiment/') 
    parser.add_argument("--save_every", type=int, help='save model every # epochs', default=5)
    parser.add_argument("--log_every", type=int, help='save logs every # batches', default=100)
    parser.add_argument("--val_every", type=int, help='validate model every # epoch', default=0) 
    parser.add_argument("--save_val_images", action='store_true', help='choose if you want to save validation images')
    parser.add_argument("--save_train_images", action='store_true', help='choose if you want to save train images')
    parser.add_argument("--wandb_dir", type=str, help="directory where to save wandb data locally", default='./wandb')
    parser.add_argument("--skip_log", action='store_true', help='choose if you want to stop wandb monitoring')
    return parser


def save_model(network, epoch, optimizer, 
               loss, best_val_loss, args,
               discriminator=None, optimizer_disc=None, 
               video_discriminator=None, optimizer_video_discriminator=None,
               best_model=False):
        
    print('Saving model epoch: ', epoch)
    if not Path(args.checkpoints_dir).exists():
        Path(args.checkpoints_dir).mkdir(exist_ok=True)

    if type(epoch) == str:
        save_dir = Path(args.checkpoints_dir, epoch + '.pt') 
    else:
        save_dir = Path(args.checkpoints_dir, 'GAN_epoch_%03d.pt' % epoch) 

    if args.use_discriminator:
        torch.save({'epoch': epoch, 
                    'model_state_dict': network.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_disc_state_dict': optimizer_disc.state_dict(),
                    'loss': loss, 
                    'best_val_loss': best_val_loss}, 
                   save_dir)
                   
    elif args.use_video_discriminator:
        torch.save({'epoch': epoch, 
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'video_discriminator_state_dict': video_discriminator.state_dict(),
                    'video_optimizer_disc_state_dict': optimizer_video_discriminator.state_dict(),
                    'loss': loss, 
                    'best_val_loss': best_val_loss}, 
                   save_dir)

    else:
        torch.save({'epoch': epoch, 
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, 
                    'best_val_loss': best_val_loss}, 
                   save_dir)

    return save_dir


def print_progress(loss, best_loss = None, mode='train'):
    print(mode + ' loss: ', loss)
    if best_loss:
        print('best ' + mode + ' loss: ', best_loss)


def set_requires_grad(network, requires_grad_flag):
    for param in network.parameters():
        param.requires_grad = requires_grad_flag


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


if __name__=='__main__':

    parser = config_parser()
    args = parser.parse_args()

    # TODO: remove after sweep
    # make new checkpoint with time  
    # ct = datetime.datetime.now()
    # args.checkpoints_dir = args.checkpoints_dir + '_' + '%02d'%ct.day + '%02d'%ct.month + '%04d'%ct.year + '_' + '%02d'%ct.hour + '%02d'%ct.minute + '%02d'%ct.second
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # copy config in checkpoint folder
    print_options(args, parser)

    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # prepare data
    train_loader = build_dataloader(args, mode='train', shuffle=False)
    if args.val_every != 0:
        val_loader = build_dataloader(args, mode='val', shuffle=False)
    else:
        val_loader = None

    # build the network 
    network = UNet(args).to(device)
    init_weights(network)

    # make the networks multi-gpu available
    if torch.cuda.device_count() > 1:
        print("There are", torch.cuda.device_count(), "gpus available.")
        network = nn.DataParallel(network)

    # Build loss function
    loss_fn = Loss(args, device).to(device)
    loss_fn.eval() # make sure networks used in calculating loss (e.g. VGG) are in eval mode (because of batch normalization and dropout layers)

    # Build metrics functions
    metrics_fn = Metrics(args, device)
    metrics_fn.reset()

    # Build an optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, betas=(0.5, 0.999))
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()
    
    # Learning rate decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=0.1)

    # Print number of parameters in the generator network
    pytorch_generator_params = sum(p.numel() for p in network.parameters())
    print('Number of parameters in the generator network: ', pytorch_generator_params)

    # build discriminator if it is used
    if args.use_discriminator:
        discriminator = NLayerDiscriminator(input_nc=args.num_input_channels+args.num_output_channels, n_layers=3).to(device)
        init_weights(discriminator)
        set_requires_grad(discriminator, False)
        loss_gan_fn = GANLoss().to(device)
        if args.optimizer == 'adam':
            optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
        elif args.optimizer == 'sgd':
            optimizer_discriminator = torch.optim.SGD(discriminator.parameters(), lr=args.lr_disc)
        pytorch_discriminator_params = sum(p.numel() for p in discriminator.parameters())
        print('Number of parameters in the discriminator network: ', pytorch_discriminator_params)
        # Learning rate decay
        lr_scheduler_discr = torch.optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=args.lr_decay_steps, gamma=0.1)

    else:
        discriminator = None
        optimizer_discriminator = None
        lr_scheduler_discr = None
    
    # build video discriminator if it is used
    if args.use_video_discriminator:
        video_discriminator = Discriminator_3D(d_conv_dim=96).to(device) # TODO: test
        init_weights(video_discriminator)
        set_requires_grad(video_discriminator, False)
        loss_gan_fn = GANLoss().to(device)
        if args.optimizer == 'adam':
            optimizer_video_discriminator = torch.optim.Adam(video_discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
        elif args.optimizer == 'sgd':
            optimizer_video_discriminator = torch.optim.SGD(video_discriminator.parameters(), lr=args.lr_disc)
        pytorch_video_discriminator_params = sum(p.numel() for p in video_discriminator.parameters())
        print('Number of parameters in the video discriminator network: ', pytorch_video_discriminator_params)
        # Learning rate decay
        lr_scheduler_video_discr = torch.optim.lr_scheduler.StepLR(optimizer_video_discriminator, step_size=args.lr_decay_steps, gamma=1) # gamma=1 for no updates in lr

    else:
        video_discriminator = None
        optimizer_video_discriminator = None
        lr_scheduler_video_discr = None
    
    best_val_loss = None

    # initialize wandb
    if not args.skip_log:
        wandb.init(project='GAN-video-synthesis', config=args, tags=["demo"], job_type='train', dir=args.wandb_dir)
        args = wandb.config
        wandb.watch(network)
    
    # Start the training process
    start_train_disc = -1
    step = 0
    for epoch in tqdm(range(args.continue_from_epoch, args.num_epochs), desc='Epoch: '):
        # print('Training epoch: ', epoch)
        network.train()
        if args.use_discriminator:
            discriminator.train()

        # Training loop
        for batch_idx, data in enumerate(train_loader):

            inputs = data["input_image"].to(device)
            labels = data["output_image"].to(device)

            if args.use_label_maps:
                label_maps = data["label_image"].to(device)
            else:
                label_maps = None

            prediction = network(inputs)

            loss_perceptual = loss_fn(prediction, labels)

            loss = loss_perceptual

            if args.use_discriminator and epoch>start_train_disc:

                set_requires_grad(discriminator, True)

                real_pair = torch.cat((inputs, labels), 1)
                pred_real = discriminator(real_pair, label_maps)
                loss_real = loss_gan_fn(pred_real, real_image_flag=True)
                
                fake_pair = torch.cat((inputs, prediction.detach()), 1)
                pred_fake = discriminator(fake_pair, label_maps)
                loss_fake = loss_gan_fn(pred_fake, real_image_flag=False)

                loss_discriminator = (loss_fake + loss_real)*0.5

                optimizer_discriminator.zero_grad()
                loss_discriminator.backward()
                optimizer_discriminator.step()
                
                set_requires_grad(discriminator, False)
                
                fake_pair = torch.cat((inputs, prediction), 1)
                pred_fake = discriminator(fake_pair, label_maps)

                loss_gan = loss_gan_fn(pred_fake, real_image_flag=True)
                loss += loss_gan 
            
            if args.use_video_discriminator and epoch>start_train_disc:
                
                # Train discriminator
                if (batch_idx % args.steps_train_video_discr) == 0:
                    set_requires_grad(video_discriminator, True)

                    pred_real = video_discriminator(labels.unsqueeze(0))
                    loss_video_real = loss_gan_fn(pred_real, real_image_flag=True)
                    
                    pred_fake = video_discriminator(prediction.unsqueeze(0).detach())
                    loss_video_fake = loss_gan_fn(pred_fake, real_image_flag=False)

                    loss_video_discriminator = (loss_video_real + loss_video_fake)*0.5

                    # print(loss_video_discriminator.item(), loss_video_fake.item(), loss_video_real.item())

                    optimizer_video_discriminator.zero_grad()
                    loss_video_discriminator.backward()
                    optimizer_video_discriminator.step()
                
                
                # Train generator
                set_requires_grad(video_discriminator, False)
                
                pred_fake = video_discriminator(prediction.unsqueeze(0).detach())
                loss_gan_video = loss_gan_fn(pred_fake, real_image_flag=True)
                # print('Perceptual loss: ', loss_perceptual.item())
                # print('GAN video loss: ', loss_gan_video.item())

                loss += loss_gan_video * args.loss_gan_video_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = step + 1

            if not args.skip_log and batch_idx % args.log_every == 0:
                if args.save_train_images:
                    image_concat = create_image_pair([inputs.detach(), prediction.detach(), labels.detach()])
                    wandb_images = [wandb.Image(image) for image in image_concat]
                    wandb.log({'train images predictions': wandb_images}, step=step)  

                wandb.log({'train_loss': loss.detach()}, step=step)
                wandb.log({'train loss perceptual': loss_perceptual.detach()}, step=step)

                if args.use_discriminator and epoch>start_train_disc:
                    wandb.log({ 'train loss gan': loss_gan, 
                                'discriminator train loss': loss_discriminator.detach(), 
                                'discriminator loss real': loss_real,
                                'discriminator loss fake': loss_fake}, step=step)
                
                if args.use_video_discriminator and epoch>start_train_disc:
                    try:
                        wandb.log({ 'train loss gan video': loss_gan_video, 
                                    'video discriminator train loss': loss_video_discriminator.detach(), 
                                    'video discriminator loss real': loss_video_real,
                                    'video discriminator loss fake': loss_video_fake}, step=step)
                    except NameError:
                         wandb.log({ 'train loss gan video': loss_gan_video}, step=step)

            if step == args.num_steps:
                print(f'Breaking at {args.num_steps} steps.')
                save_dir = save_model(network, 'latest_GAN_model', optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, video_discriminator, optimizer_video_discriminator, best_model=False)
                # model_artifact = wandb.Artifact('last_model_train', type='model')
                # model_artifact.add_file(save_dir)
                # wandb.log_artifact(model_artifact)
                exit()
            
        # Save model
        if epoch % args.save_every == 0 and epoch > 0:
            save_dir = save_model(network, epoch, optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, video_discriminator, optimizer_video_discriminator, best_model=False)
            # model_artifact = wandb.Artifact('last_model_train', type='model')
            # model_artifact.add_file(save_dir)
            # wandb.log_artifact(model_artifact)
        
        # Update learning rates
        lr_scheduler.step()
        if args.use_discriminator:
            lr_scheduler_discr.step()
        if args.use_video_discriminator:
            lr_scheduler_video_discr.step()
        
        # Validation
        if args.val_every != 0 and epoch % args.val_every == 0 and epoch > 0:
            print('Validating epoch: ', epoch)
            val_loss = 0
            count = 0
            metrics_fn.reset()

            with torch.no_grad():
                network.eval()
                for batch_idx, data in enumerate(val_loader):

                    inputs = data["input_image"].to(device)
                    labels = data["output_image"].to(device)

                    prediction = network(inputs)

                    # compute metrics
                    metrics_fn.update(prediction, labels)

                    # compute losses
                    loss_perceptual_val = loss_fn(prediction, labels)
                    loss_val = loss_perceptual_val

                    if args.use_video_discriminator and epoch>start_train_disc:

                        pred_real_val = video_discriminator(labels.unsqueeze(0))
                        loss_video_real_val = loss_gan_fn(pred_real_val, real_image_flag=True)
                        
                        pred_fake_val = video_discriminator(prediction.unsqueeze(0))
                        loss_video_fake_val = loss_gan_fn(pred_fake_val, real_image_flag=False)

                        loss_video_discriminator_val = (loss_video_real_val + loss_video_fake_val)*0.5

                        loss_gan_video_val = loss_gan_fn(pred_fake_val, real_image_flag=True)
        
                        loss_val += loss_gan_video_val * args.loss_gan_video_weight
                    
                    val_loss += loss_val.detach()
                    count += 1

            avg_val_loss = val_loss/count

            metrics_log, metrics_combined = metrics_fn.compute()

            if not args.skip_log:
                if args.save_val_images:
                    image_concat = create_image_pair([inputs.detach(), prediction.detach(), labels.detach()])
                    wandb_images = [wandb.Image(image) for image in image_concat]
                    wandb.log({'val loss': avg_val_loss, 'validation images predictions': wandb_images}, step=step)
                    wandb.log(metrics_log, step=step)
                    wandb.log({'metrics_combined': metrics_combined}, step=step)

                else:   
                    wandb.log({'val loss': avg_val_loss}, step=step)
                    wandb.log(metrics_log, step=step)
                    wandb.log({'metrics_combined': metrics_combined}, step=step)

    save_dir = save_model(network, 'latest_GAN_model', optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, video_discriminator, optimizer_video_discriminator, best_model=False)
    # model_artifact = wandb.Artifact('last_model_train', type='model')
    # model_artifact.add_file(save_dir)
    # wandb.log_artifact(model_artifact)
