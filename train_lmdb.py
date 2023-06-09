from dataloaders.EnMAP_dataset_lmdb import enmap_dataset

import os
import argparse
import json
import torch
# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
# import kornia
# from kornia import laplacian, sobel
from scipy.io import savemat
import torch.nn.functional as F
from utils.vgg_perceptual_loss import VGGPerceptualLoss, VGG19
from utils.hyper_perceptual_loss import HyperPerceptualLoss
from utils.spatial_loss import Spatial_Loss



# num_channels = 102
config_path = r'/vol/research/RobotFarming/Projects/hyper_transformer/configs/config_HSIT_botswana_ft_vgg_1.json'
best_pre_model_path = r'/vol/research/RobotFarming/Projects/hyper_transformer/bst_model/botswana_pre_best_model.pth'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
def read_csv_keys(csv_file, csv_file_col_name):
    df = pd.read_csv(csv_file)
    keys = df[csv_file_col_name].tolist()
    return keys

__dataset__ = {
    "pavia_dataset": pavia_dataset, 
    "botswana_dataset": botswana_dataset, 
    "chikusei_dataset": chikusei_dataset, 
    "botswana4_dataset": botswana4_dataset,
    "enmap_dataset": enmap_dataset
}

# Parse the arguments
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default=config_path,type=str,
                        help='Path to the config file')
parser.add_argument('-r', '--resume', default=best_pre_model_path, type=str,
                        help='Path to the .pth model checkpoint to resume training')
parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
parser.add_argument('--local', action='store_true', default=False)
args = parser.parse_args()

# Loading the config file
config = json.load(open(args.config))
torch.backends.cudnn.benchmark = True
# breakpoint()
# Set seeds.
torch.manual_seed(7)

# Setting number of GPUS available for training.
num_gpus = torch.cuda.device_count()

# Selecting the model.
model = MODELS[config["model"]](config)
# print(f'\n{model}\n')

# Sending model to GPU  device.
if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()

# Setting up training and testing dataloaderes.
print("Training with dataset => {}".format(config["train_dataset"]))
if config["train_dataset"] == 'enmap_dataset':
    keys = read_csv_keys(
                        os.path.join(config["enmap_dataset"]["lmdb_save_dir"], 
                                        config["enmap_dataset"]["csv_file_name"]), 
                        config["enmap_dataset"]["columns"][0])

    train_keys, val_keys = train_test_split(
                                            keys, 
                                            test_size=config["enmap_dataset"]["val_split"], 
                                            random_state=42, shuffle=False)

    print(f"Number of train set: {len(train_keys)} && Number of validation set: {len(val_keys)}")
    train_loader = data.DataLoader(
                            __dataset__[config["train_dataset"]](
                                config,
                                train_keys,
                                is_train=True,
                                want_DHP_MS_HR=config["is_DHP_MS"],
                            ),
                            batch_size=config["train_batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=True,
                            pin_memory=False,
                        )

    test_loader = data.DataLoader(
                            __dataset__[config["train_dataset"]](
                                config,
                                val_keys,
                                is_train=False,
                                want_DHP_MS_HR=config["is_DHP_MS"],
                            ),
                            batch_size=config["val_batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=True,
                            pin_memory=False,
                        )
else:
    train_loader = data.DataLoader(
                        __dataset__[config["train_dataset"]](
                            config,
                            is_train=True,
                            want_DHP_MS_HR=config["is_DHP_MS"],
                        ),
                        batch_size=config["train_batch_size"],
                        num_workers=config["num_workers"],
                        shuffle=True,
                        pin_memory=False,
                    )

    test_loader = data.DataLoader(
                            __dataset__[config["train_dataset"]](
                                config,
                                is_train=False,
                                want_DHP_MS_HR=config["is_DHP_MS"],
                            ),
                            batch_size=config["val_batch_size"],
                            num_workers=config["num_workers"],
                            shuffle=True,
                            pin_memory=False,
                        )

# Initialization of hyperparameters. 
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]

# Setting up optimizer.
if config["optimizer"]["type"] == "SGD":
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["optimizer"]["args"]["lr"], 
        momentum = config["optimizer"]["args"]["momentum"], 
        weight_decay= config["optimizer"]["args"]["weight_decay"]
    )
elif config["optimizer"]["type"] == "ADAM":
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["optimizer"]["args"]["lr"],
        weight_decay= config["optimizer"]["args"]["weight_decay"]
    )
else:
    exit("Undefined optimizer type")

# Learning rate sheduler. 
scheduler = optim.lr_scheduler.StepLR(  optimizer, 
                                        step_size=config["optimizer"]["step_size"], 
                                        gamma=config["optimizer"]["gamma"])

# Resume...
if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    checkpoint = torch.load(args.resume)
    # model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint, strict=False)
else:
    # initialize_weights(model)
    initialize_weights_new(model)

# Setting up loss functions.
if config[config["train_dataset"]]["loss_type"] == "L1":
    criterion   = torch.nn.L1Loss()
    HF_loss     = torch.nn.L1Loss()
elif config[config["train_dataset"]]["loss_type"] == "MSE":
    criterion   = torch.nn.MSELoss()
    HF_loss     = torch.nn.MSELoss()
else:
    exit("Undefined loss type")

if config[config["train_dataset"]]["VGG_Loss"]:
    vggnet = VGG19()
    vggnet = torch.nn.DataParallel(vggnet).cuda()

if config[config["train_dataset"]]["Spatial_Loss"]:
    Spatial_loss = Spatial_Loss(in_channels = config[config["train_dataset"]]["spectral_bands"]).cuda()

# Training epoch.
def train(epoch):
    
    train_loss = 0.0
    model.train()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader, 0):
        # Reading data.
        _, MS_image, PAN_image, reference = data

        # Making Smaller Patches for the training
        if config["trainer"]["is_small_patch_train"]:
            MS_image,_ = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
            PAN_image,_ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
            reference,_ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

        # Taking model outputs ...
        MS_image    = Variable(MS_image.float().cuda()) 
        PAN_image   = Variable(PAN_image.float().cuda()) 
        out         = model(MS_image, PAN_image)

        outputs = out["pred"]

        ######### Computing loss #########
        # Normal L1 loss
        if config[config["train_dataset"]]["Normalized_L1"]:
            max_ref     = torch.amax(reference, dim=(2,3)).unsqueeze(2).unsqueeze(3).expand_as(reference).cuda()
            loss        = criterion(outputs/max_ref, to_variable(reference)/max_ref)
        else:
            loss        = criterion(outputs, to_variable(reference))

        # print(f'L1 Loss: {loss}')
        # VGG Perceptual Loss
        if config[config["train_dataset"]]["VGG_Loss"]:
            predicted_RGB   = torch.cat((torch.mean(outputs[:, config[config["train_dataset"]]["R"]-2:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["G"]-2:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["B"]-2:config[config["train_dataset"]]["B"], :, :], 1).unsqueeze(1)), 1)
            
            target_RGB   = torch.cat((torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["R"]-2:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["G"]-2:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(to_variable(reference)[:, config[config["train_dataset"]]["B"]-2:config[config["train_dataset"]]["B"], :, :], 1).unsqueeze(1)), 1)
            VGG_loss        = VGGPerceptualLoss(predicted_RGB, target_RGB, vggnet)
            loss            += config[config["train_dataset"]]["VGG_Loss_F"]*VGG_loss
            # print(f'VGG Loss: {VGG_loss}')
        # Hyperspectral Perceptual Loss
        if config[config["train_dataset"]]["HVGG_Loss"]:
            model_name = config["backbone"]["model_name"] 
            pretrained = config["backbone"]["pretrained"] 
            in_channels = config["backbone"]["in_channels"]
            out_features = config["backbone"]["out_features"] 
            normalize = config["backbone"]["normalize"]
            saved_model_path = config["backbone"]["saved_model_path"]

            if config["train_dataset"] == 'enmap_dataset':
                mean_file = config["enmap_dataset"]["l1_mean_file"] # l2_mean_file l1_mean_file
                std_file = config["enmap_dataset"]["l1_std_file"] # l2_std_file l1_std_file
            else:
                mean_file = None
                std_file = None
            
            predicted_HYP   = outputs
            target_HYP   = to_variable(reference)
            # print(f'model_name: {model_name}')
            hyper_criterion  = HyperPerceptualLoss(model_name=model_name, pretrained=pretrained, in_channels=in_channels,
                                                   out_features=out_features, saved_model_path=saved_model_path, 
                                                   mean_file=mean_file, std_file=std_file, normalize=normalize)
            
            hyper_loss        = hyper_criterion(predicted_HYP, target_HYP)
            loss            += config[config["train_dataset"]]["HVGG_Loss_F"]*hyper_loss
            # print(f'Hyper Loss: {hyper_loss}')

        # Transfer Perceptual Loss
        if config[config["train_dataset"]]["Transfer_Periferal_Loss"]:
            loss += config[config["train_dataset"]]["Transfer_Periferal_Loss_F"]*out["tp_loss"]

        # Spatial loss
        if config[config["train_dataset"]]["Spatial_Loss"]:
            loss += config[config["train_dataset"]]["Spatial_Loss_F"]*Spatial_loss(to_variable(reference), outputs)
        
        # Spatial loss
        if config[config["train_dataset"]]["multi_scale_loss"]:
            loss += config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x13"]) + 2*config[config["train_dataset"]]["multi_scale_loss_F"]*criterion(to_variable(reference), out["x23"])

        # print(f'Train Loss: {loss}')
        torch.autograd.backward(loss)
        # loss.backward()

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    writer.add_scalar('Loss/train', loss, epoch)

    # if config[config["train_dataset"]]["VGG_Loss"]:
    #     writer.add_scalar('VGG Loss/train', VGG_loss, epoch)

    # if config[config["train_dataset"]]["HVGG_Loss"]:
    #     writer.add_scalar('Hyperspectral Loss/train', hyper_loss, epoch)

    
# Testing epoch.
def test(epoch):
    test_loss   = 0.0
    cc          = 0.0
    sam         = 0.0
    rmse        = 0.0
    ergas       = 0.0
    psnr        = 0.0
    val_outputs = {}
    model.eval()
    pred_dic = {}
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            image_dict, MS_image, PAN_image, reference = data

            # Generating small patches
            if config["trainer"]["is_small_patch_train"]:
                MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
                PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
                reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            # Inputs and references...
            MS_image    = MS_image.float().cuda()
            PAN_image   = PAN_image.float().cuda()
            reference   = reference.float().cuda()

            # Taking model output
            out     = model(MS_image, PAN_image)

            outputs = out["pred"]

            # Computing validation loss
            loss        = criterion(outputs, reference)
            test_loss   += loss.item()

            # Scalling
            outputs[outputs<0]      = 0.0
            outputs[outputs>1.0]    = 1.0
            outputs                 = torch.round(outputs*config[config["train_dataset"]]["max_value"])
            img_dict = {image_dict["imgs"][0].split("/")[-1][:-4]+"_pred": torch.squeeze(outputs).permute(1,2,0).cpu().numpy()}
            pred_dic.update(img_dict)
            reference               = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])

        
            ### Computing performance metrics ###
            # Cross-correlation
            cc += cross_correlation(outputs, reference)
            # SAM
            sam += SAM(outputs, reference)
            # RMSE
            rmse += RMSE(outputs/torch.max(reference), reference/torch.max(reference))
            # ERGAS
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()
            ergas += ERGAS(outputs, reference, beta)
            # PSNR
            psnr += PSNR(outputs, reference)

    # Taking average of performance metrics over test set
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    # Writing test results to tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)

    # Images to tensorboard
    # Regenerating the final image
    if config["trainer"]["is_small_patch_train"]:
        outputs = outputs.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        outputs = outputs.contiguous().view(config["val_batch_size"], 
                                            config[config["train_dataset"]]["spectral_bands"],
                                            config[config["train_dataset"]]["HR_size"],
                                            config[config["train_dataset"]]["HR_size"])
        reference = reference.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        reference = reference.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
        MS_image = MS_image.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        MS_image = MS_image.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
    # if (epoch + 1) % 10 == 0:
    #Normalizing the images
    outputs     = outputs/torch.max(reference)
    reference   = reference/torch.max(reference)
    MS_image    = MS_image/torch.max(reference)

    if config["model"]=="HyperPNN" or config["is_DHP_MS"]==False:
        MS_image =  F.interpolate(MS_image, scale_factor=(config[config["train_dataset"]]["factor"],config[config["train_dataset"]]["factor"]),mode ='bilinear')

    r, g, b = config[config["train_dataset"]]["R"], config[config["train_dataset"]]["G"], config[config["train_dataset"]]["B"]
    ms      = MS_image[:, [r, g, b], :, :]
    pred    = outputs[:, [r, g, b], :, :]
    ref     = reference[:, [r, g, b], :, :]

    imgs    = torch.zeros(5*pred.shape[0], 3, pred.shape[2], pred.shape[3])
    for i in range(pred.shape[0]):
        imgs[5*i]   = ms[i]
        imgs[5*i+1] = torch.abs(ms[i]-pred[i])/torch.max(torch.abs(ms[i]-pred[i]))
        imgs[5*i+2] = pred[i]
        imgs[5*i+3] = ref[i]
        imgs[5*i+4] = torch.abs(ref[i]-pred[i])/torch.max(torch.abs(ref[i]-pred[i]))

    imgs = torchvision.utils.make_grid(imgs, nrow=5)
    writer.add_image('Images', imgs[:,:,:], epoch+1)

    #Return Outputs
    metrics = { "loss": float(test_loss), 
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    print(f'Test Loss: {test_loss}')
    return image_dict, pred_dic, metrics

# Setting up tensorboard and copy .json file to save directory.
PATH = f'{config["root_dir"]}/hyper_transformer/{config["experim_name"]}/{config["train_dataset"]}/{config["cfg_suffix"]}'
ensure_dir(PATH+"/")
writer = SummaryWriter(log_dir=PATH)
shutil.copy2(args.config, PATH)

# Print model to text file
original_stdout = sys.stdout 
with open(PATH+"/"+"model_summary.txt", 'w+') as f:
    sys.stdout = f
    print(f'\n{model}\n')
    sys.stdout = original_stdout 

# Main loop.
best_psnr   =0.0
print(config["cfg_suffix"])
best_model_checkpoint = os.path.join(PATH, "best_model.pth")

if os.path.exists(best_model_checkpoint):
    checkpoint = torch.load(best_model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded model checkpoint from {best_model_checkpoint}")
    print(f'starting from epoch {start_epoch}')

for epoch in range(start_epoch, total_epochs):
    print("\nTraining Epoch: %d" % epoch)
    train(epoch)

    if epoch % config["trainer"]["test_freq"] == 0:
        print("\nTesting Epoch: %d" % epoch)
        image_dict, pred_dic, metrics=test(epoch)
        
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            
            
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict()}, best_model_checkpoint)

            with open(PATH+"/"+"best_metrics.json", "w+") as outfile: 
                json.dump(metrics, outfile)

            savemat(PATH+"/"+ "final_prediction.mat", pred_dic)

    scheduler.step()

# python train.py --config configs/config_HSIT_PRE.json
# python train.py --config configs/config_HSIT.json --resume ./bst_model/pavia_pre_best_model.pth