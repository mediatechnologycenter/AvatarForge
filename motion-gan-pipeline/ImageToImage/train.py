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
    parser.add_argument("--use_label_maps", action='store_true', help="choose if to use label maps for discriminator")
    # dataset options
    parser.add_argument("--dataset_type", type=str, help="options: CustomDataset", default='CustomDataset') 
    parser.add_argument("--input_train_root_dir", type=str, help="Path to training input images", default='./data/input_images_train') 
    parser.add_argument("--output_train_root_dir", type=str, help="Path to training output images", default='./data/output_images_train')  
    parser.add_argument("--label_train_root_dir", type=str, help="Path to training label images", default='./data/label_images_train')  
    parser.add_argument("--input_val_root_dir", type=str, help="Path to val input images", default='./data/input_images_val') 
    parser.add_argument("--output_val_root_dir", type=str, help="Path to val output images", default='./data/output_images_val')
    parser.add_argument("--label_val_root_dir", type=str, help="Path to val label images", default='./data/label_images_val')  
    parser.add_argument("--width", type=int, default=640, help="width")
    parser.add_argument("--height", type=int, default=360, help="height")
    # logging/saving options
    parser.add_argument("--checkpoints_dir", type=str, help='specify the directory to save the model', default='./chkpts/experiment/') 
    parser.add_argument("--save_every", type=int, help='save model every # epochs', default=5)
    parser.add_argument("--log_every", type=int, help='save logs every # batches', default=100)
    parser.add_argument("--save_val_images", action='store_true', help='choose if you want to save validation images')
    parser.add_argument("--wandb_dir", type=str, help="directory where to save wandb data locally", default='./wandb')
    return parser


def save_model(network, epoch, optimizer, loss, best_val_loss, args, discriminator=None, optimizer_disc=None, best_model=False):
    print('Saving model epoch: ', epoch)
    if not Path(args.checkpoints_dir).exists():
        Path(args.checkpoints_dir).mkdir(exist_ok=True)

    save_dir = Path(args.checkpoints_dir, 'latest_GAN.pt') 

    if args.use_discriminator:
        torch.save({'epoch': epoch, 
                    'model_state_dict': network.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_disc_state_dict': optimizer_disc.state_dict(),
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

    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # prepare data
    train_loader = build_dataloader(args, mode='train', shuffle=True)

    # build the network 
    network = UNet(args).to(device)
    init_weights(network)

    # make the networks multi-gpu available
    if torch.cuda.device_count() > 1:
        print("There are", torch.cuda.device_count(), "gpus available.")
        network = nn.DataParallel(network)

    # Build loss function
    loss_fn = Loss(args, device).to(device)
    # loss_fn.eval() # make sure networks used in calculating loss (e.g. VGG) are in eval mode (because of batch normalization and dropout layers)

    #Build metrics functions
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
    else:
        discriminator = None
        optimizer_discriminator = None
    
    best_val_loss = None

    # initialize wandb
    wandb.init(project='GAN-video-synthesis', config=args, tags=["demo"], job_type='train', dir=args.wandb_dir)
    args = wandb.config
    wandb.watch(network)
    
    # Start the training process
    start_train_disc = -1
    step = 0
    for epoch in range(args.continue_from_epoch, args.num_epochs):
        print('Training epoch: ', epoch)

        network.train()
        if args.use_discriminator:
            discriminator.train()

        for batch_idx, data in enumerate(train_loader):

            inputs = data["input_image"].to(device)
            labels = data["output_image"].to(device)
            if args.use_label_maps:
                label_maps = data["label_image"].to(device)
            else:
                label_maps = None

            prediction = network(inputs)

            loss_perceptual = loss_fn(prediction, labels)

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

                loss = loss_perceptual + loss_gan
            else:
                loss = loss_perceptual
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = step + 1

            if batch_idx % args.log_every == 0:
                if args.save_val_images:   
                    image_concat = create_image_pair([inputs.detach(), prediction.detach(), labels.detach()])
                    wandb_images = [wandb.Image(image) for image in image_concat]
                    if args.use_discriminator and epoch>start_train_disc:
                        wandb.log({'train loss perceptual': loss_perceptual, 
                                   'train loss gan': loss_gan, 
                                   'discriminator train loss': loss_discriminator.detach(), 
                                   'train images predictions': wandb_images,
                                   'discriminator loss real': loss_real,
                                   'discriminator loss fake': loss_fake}, step=step)
                        wandb.log({'train_loss': loss.detach()}, step=step)
                    else:
                        wandb.log({'train images predictions': wandb_images}, step=step)                        
                        wandb.log({'train_loss': loss.detach()}, step=step)
                else:
                    if args.use_discriminator and epoch>start_train_disc:
                        wandb.log({'train loss perceptual': loss_perceptual,
                                   'train loss gan': loss_gan,
                                   'discriminator train loss': loss_discriminator.detach(),
                                   'discriminator loss real': loss_real,
                                   'discriminator loss fake': loss_fake}, step=step)
                        wandb.log({'train_loss': loss.detach()}, step=step)
                    else:
                        wandb.log({'train_loss': loss.detach()}, step=step)                      
            
            if step == args.num_steps:
                print(f'Breaking at {args.num_steps} steps.')
                save_dir = save_model(network, epoch, optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, best_model=False)
                model_artifact = wandb.Artifact('last_model_train', type='model')
                model_artifact.add_file(save_dir)
                wandb.log_artifact(model_artifact)
                exit()
        
        if epoch % args.save_every == 0:
            save_dir = save_model(network, epoch, optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, best_model=False)
            model_artifact = wandb.Artifact('last_model_train', type='model')
            model_artifact.add_file(save_dir)
            wandb.log_artifact(model_artifact)
        
        lr_scheduler.step()

    save_dir = save_model(network, epoch, optimizer, loss, best_val_loss, args, discriminator, optimizer_discriminator, best_model=False)
    model_artifact = wandb.Artifact('last_model_train', type='model')
    model_artifact.add_file(save_dir)
    wandb.log_artifact(model_artifact)