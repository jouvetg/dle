#!/usr/bin/env python3

"""
dle.py contains all functions for the 2Dto2D-DLE (Deep Learning Emulator)
@author: Guillaume Jouvet
"""

import numpy as np
import os, sys, shutil, glob, math
import matplotlib.pyplot as plt
import datetime, time
import argparse
from   netCDF4 import Dataset
import tensorflow as tf
from scipy import interpolate

def str2bool(v):
    return v.lower() in ("true", "1")

class dle:

    ####################################################################################
    #                                 INITIALIZATION
    ####################################################################################

    def __init__(self):
        """
            initialize the class DLE
        """

        self.parser = argparse.ArgumentParser(description="2Dto2D-DLE")

        self.read_config_param()

        self.config = self.parser.parse_args()
        
        self.set_pre_defined_mapping()

    def read_config_param(self):
        """
            read config file
        """

        self.parser.add_argument(
            "--dataset",
            type=str,
            default="pism_isoth_2000",
            help="Name of the dataset used for training",
        )
        self.parser.add_argument(
            "--maptype",
            type=str,
            default="f2",
            help="This use a predefined mapping, otherwise it is the folder name where to find fieldin.dat and fieldout.dat defining the mapping",
        )   
        self.parser.add_argument(
            "--train",
            type=str2bool,
            default=True,
            help="Set this to True if you wish to train",
        )
        self.parser.add_argument(
            "--predict",
            type=str2bool,
            default=True,
            help="Set this to True if you wish to predict",
        )
        self.parser.add_argument(
            "--network", 
            type=str, 
            default="cnn", 
            help="This is the type of network, it can be cnn or unet"
        )
        self.parser.add_argument(
            "--conv_ker_size", 
            type=int, 
            default=3, 
            help="Convolution kernel size for CNN"
        )
        self.parser.add_argument(
            "--nb_layers",
            type=int,
            default=16,
            help="Number of convolutional layers in the CNN",
        )
        self.parser.add_argument(
            "--nb_blocks",
            type=int,
            default=4,
            help="Number of block layer in the U-net",
        )
        self.parser.add_argument(
            "--nb_out_filter",
            type=int,
            default=32,
            help="Number of filters in the CNN or Unet",
        )
        self.parser.add_argument(
            "--activation",
            type=str,
            default="lrelu",
            help="Neural network activation function (lrelu = LeaklyRelu)",
        )
        self.parser.add_argument(
            "--dropout_rate",
            type=float,
            default=0,
            help="Neural network Drop out rate",
        )
        self.parser.add_argument(
            "--learning_rate", 
            type=float, 
            default=0.0001, 
            help="Learning rate"
        )
        self.parser.add_argument(
            "--clipnorm",
            type=float,
            default=0.5,
            help="Parameter that can clip the gradient norm (0.5)",
        )
        self.parser.add_argument(
            "--regularization",
            type=float,
            default=0.0,
            help="Regularization weight (0)",
        )
        self.parser.add_argument(
            "--batch_size", 
            type=int, 
            default=64, 
            help="Batch size for training (64)"
        )
        self.parser.add_argument(
            "--loss", 
            type=str, 
            default="mae", 
            help="Type of loss : mae or mse"
        )
        self.parser.add_argument(
            "--resample_data",
            type=float,
            default=1.0,
            help="Coarsen data by averaging to generate other resolution",
        )
        self.parser.add_argument(
            "--include_test_in_train",
            type=str2bool,
            default=False,
            help="Force including the test data in the training dataset",
        )
        self.parser.add_argument(
            "--save_model_each",
            type=int,
            default=10000,
            help="The model is save each --save_model_each epochs",
        )
        self.parser.add_argument(
            "--data_augmentation", 
            type=str2bool, 
            default=True, 
            help="Augment data with some transformation"
        )
        self.parser.add_argument(
            "--data_stepping", 
            type=int, 
            default=1, 
            help="This serves to take only a small part of data with a given step, practical for quick test"
        )
        self.parser.add_argument(
            "--epochs", 
            type=int, 
            default=100, 
            help="Number of epochs, i.e. pass over the data set at training"
        )
        self.parser.add_argument(
            "--seed", 
            type=int, 
            default=123, 
            help="Seed ID for reproductability"
        )
        self.parser.add_argument(
            "--thrice",
            type=float,
            default=1,
            help="Threshold for the presence of ice to exclude ice-free patches",
        )
        self.parser.add_argument(
            "--patch_size",
            type=int,
            default=32,
            help="Patch size if negative then it compute the highest possible ",
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="/home/jouvetg/DLE/data",
            help="Path of the data folder",
        )
        self.parser.add_argument(
            "--results_dir",
            type=str,
            default="/home/jouvetg/DLE/results",
            help="Path of the results folder",
        )
        self.parser.add_argument(
            "--verbose", 
            type=int, 
            default=1, 
            help="Verbosity level at training (1)"
        )
        self.parser.add_argument(
            "--usegpu",
            type=str2bool,
            default=True,
            help="use the GPU at training, this is nearly mandatory",
        )
        
    def set_pre_defined_mapping(self):

        self.mappings = {}

        self.mappings["f2"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f10"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "strflowctrl"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f11"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "arrhenius","slidingco"],
            "fieldout": ["ubar", "vbar"],
        }

        self.mappings["f12"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "strflowctrl"],
            "fieldout": ["ubar", "vbar", "uvelsurf", "vvelsurf"],
        }

        self.mappings["f13"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "arrhenius","slidingco"],
            "fieldout": ["ubar", "vbar", "uvelsurf", "vvelsurf"],
        }

        self.mappings["f14"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "strflowctrl"],
            "fieldout": ["uvelbase","vvelbase","ubar","vbar","uvelsurf","vvelsurf"],
        }

        self.mappings["f15"] = {
            "fieldin": ["thk", "slopsurfx", "slopsurfy", "arrhenius","slidingco"],
            "fieldout": ["uvelbase","vvelbase","ubar","vbar","uvelsurf","vvelsurf"],
        }

        self.naturalbounds = {}
        self.naturalbounds["thk"]         = 2000.0
        self.naturalbounds["slopsurfx"]   = 1.5
        self.naturalbounds["slopsurfy"]   = 1.5
        self.naturalbounds["slopsurfn"]   = 1.5
        self.naturalbounds["ubar"]        = 1000.0
        self.naturalbounds["vbar"]        = 1000.0
        self.naturalbounds["uvelsurf"]    = 1000.0
        self.naturalbounds["vvelsurf"]    = 1000.0
        self.naturalbounds["uvelbase"]    = 1000.0
        self.naturalbounds["vvelbase"]    = 1000.0
        self.naturalbounds["strflowctrl"] = 200.0
        self.naturalbounds["arrhenius"]   = 100.0
        self.naturalbounds["slidingco"]   = 100.0

    def initialize(self):
        """
            function initialize the absolute necessary
        """

        print(
            " -------------------- START 2Dto2D DLE --------------------------"
        )

        if self.config.network == "unet":
            self.network_depth = self.config.nb_blocks
        else:
            self.network_depth = self.config.nb_layers

        if not self.config.resample_data == 1:
            new_reso = int(
                self.config.resample_data * int(self.config.dataset.split("_")[-1])
            )
            ln = self.config.dataset.split("_")
            ln[-1] = str(new_reso)
            ndataset = "_".join(ln)
        else:
            ndataset = self.config.dataset

        self.pathofresults = os.path.join(
            self.config.results_dir,
            ndataset
            + "+"
            + "MAPTY-"
            + self.config.maptype
            + "+"
            + "NETWO-"
            + self.config.network
            + "+"
            + "CONVS-"
            + str(self.config.conv_ker_size)
            + "+"
            + "DEPTH-"
            + str(self.network_depth)
            + "+"
            + "NBFIL-"
            + str(self.config.nb_out_filter)
            + "+"
            + "EPOCH-"
            + str(self.config.epochs)
            + "+"
            + "RSAMP-"
            + str(self.config.resample_data)     
            + "+"
            + "TESTI-"
            + str(int(self.config.include_test_in_train)),
        )

        self.modelfile = os.path.join(self.pathofresults, "model.h5")

        # creat fieldin, fieldout, fieldbound
        if self.config.train:
            if self.config.maptype in self.mappings.keys():               
                mapping = self.mappings[self.config.maptype]
                self.fieldin = mapping["fieldin"]
                self.fieldout = mapping["fieldout"]
                self.get_field_bounds()
            else:
                self.read_fields_and_bounds(self.config.maptype)
        else:
            self.read_fields_and_bounds(self.pathofresults)

        # define the device to make computations (CPU or GPU)
        self.device_name = "/GPU:0" * self.config.usegpu + "/CPU:0" * (
            not self.config.usegpu
        )

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

        if self.config.usegpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1" for multiple
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("-------------------- LIST of PARAMS --------------------------")
        for ck in self.config.__dict__:
            print("%30s : %s" % (ck, self.config.__dict__[ck]))

        print("----------------- CREATE PATH OF RESULTS ------------------------")

        if self.config.train:

            if os.path.exists(self.pathofresults):
                shutil.rmtree(self.pathofresults)

            os.makedirs(self.pathofresults, exist_ok=True)

            self.print_fields_in_out(self.pathofresults)

            with open(os.path.join(self.pathofresults, "dle-run-parameters.txt"), "w") as f:
                for ck in self.config.__dict__:
                    print("%30s : %s" % (ck, self.config.__dict__[ck]), file=f)

    def get_field_bounds(self):
        """
            get the fieldbounds (or scaling) from predfined values
        """

        self.fieldbounds = {}

        for f in self.fieldin:
            if f in self.naturalbounds.keys():
                self.fieldbounds[f] = self.naturalbounds[f]

        for f in self.fieldout:
            if f in self.naturalbounds.keys():
                self.fieldbounds[f] = self.naturalbounds[f]

    def read_fields_and_bounds(self, path):
        """
            get fields (input and outputs) from given file
        """

        self.fieldbounds = {}
        self.fieldin = []
        self.fieldout = []

        fid = open(os.path.join(path, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldin.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            self.fieldout.append(part[0])
            self.fieldbounds[part[0]] = float(part[1])
        fid.close()

    def print_fields_in_out(self, path):
        """
            print field inputs and outputs togther with field bounds (scaling)
        """

        fid = open(os.path.join(path, "fieldin.dat"), "w")
        for key in self.fieldin:
            fid.write("%s %.6f \n" % (key, self.fieldbounds[key]))
        fid.close()

        fid = open(os.path.join(path, "fieldout.dat"), "w")
        for key in self.fieldout:
            fid.write("%s %.6f \n" % (key, self.fieldbounds[key]))
        fid.close()

    ####################################################################################
    #                                 OPENING DATA
    #################################################################################### 

    def findsubdata(self, folder):
        """
            find the directory of 'test' and 'train' folder, to reference data
        """

        subdatasetpath = [f.path for f in os.scandir(folder) if f.is_dir()]

        subdatasetpath.sort(key=lambda x: (os.path.isdir(x), x))  # sort alphabtically

        subdatasetname = [f.split("/")[-1] for f in subdatasetpath]

        return subdatasetname, subdatasetpath

    def add_slope_to_vars(self,allvar):
    
        allvar["slopsurfx"] = np.zeros_like(allvar["usurf"])
        allvar["slopsurfy"] = np.zeros_like(allvar["usurf"])
        dX = allvar["x"][1] - allvar["x"][0]
        dY = allvar["y"][1] - allvar["y"][0]
        for j in range(len(allvar["slopsurfx"])):
            allvar["slopsurfx"][j] = np.gradient(allvar["usurf"][j], dX, axis=1)
            allvar["slopsurfy"][j] = np.gradient(allvar["usurf"][j], dY, axis=0)
            
    def add_xy_to_vars(self,allvar):
    
        allvar["X"] = np.zeros_like(allvar["usurf"])
        allvar["Y"] = np.zeros_like(allvar["usurf"])
        for j in range(len(allvar["slopsurfx"])):
            allvar["X"][j], allvar["Y"][j] = np.meshgrid(allvar["x"], allvar["y"])

    def open_dataset(self, subdatasetpath, typefile):
        """
            open data assuming netcdf format
        """

        r = self.config.data_stepping

        DATAIN = []
        DATAOUT = []

        for sf in subdatasetpath:

            print("Opening data : ", sf)

            if typefile == "nc":

                nc = Dataset(os.path.join(sf, "ex.nc"), "r")

                allvar = {}
                for f in nc.variables:
                    allvar[f] = np.squeeze(nc.variables[f]).astype("float32")
                    
                self.add_slope_to_vars(allvar)
                self.add_xy_to_vars(allvar)

                NPDATAIN = np.stack([allvar[f][::r] for f in self.fieldin], axis=-1)
                NPDATAOUT = np.stack([allvar[f][::r] for f in self.fieldout], axis=-1)
                
                nc.close()

            elif typefile == "npy":

                NPDATAIN = np.stack(
                    [np.load(os.path.join(sf, f + ".npy"))[::r] for f in self.fieldin],
                    axis=-1,
                )
                NPDATAOUT = np.stack(
                    [np.load(os.path.join(sf, f + ".npy"))[::r] for f in self.fieldout],
                    axis=-1,
                )

            assert not np.any(np.isnan(NPDATAIN))
            assert not np.any(np.isnan(NPDATAOUT))

            DATAIN.append(NPDATAIN)
            DATAOUT.append(NPDATAOUT)
            
            del NPDATAIN,NPDATAOUT

        return DATAIN, DATAOUT

    # def resample_data(self, DATA, k):

    #     DATAUP = []

    #     # upsample by averaging
    #     if k > 1:
    #         kk = int(k)
    #         for mat in DATA:
    #             t, m, n, o = mat.shape
    #             ny = m // kk
    #             nx = n // kk
    #             DATAUP.append(
    #                 np.nanmean(
    #                     mat[:, : ny * kk, : nx * kk, :].reshape((t, ny, kk, nx, kk, o)),
    #                     axis=(2, 4),
    #                 )
    #             )

    #     # downsample by cubic interpolation
    #     elif k < 1:
 
    #         for mat in DATA:
    #             y = np.arange(mat.shape[1])
    #             x = np.arange(mat.shape[2])
    #             ynew = np.arange(int(mat.shape[1] / k)) * k
    #             ynew = ynew[ynew <= max(y)]
    #             xnew = np.arange(int(mat.shape[2] / k)) * k
    #             xnew = xnew[xnew <= max(x)]
    #             mato = np.zeros((mat.shape[0], len(ynew), len(xnew), mat.shape[3]))
    #             for m in range(mat.shape[0]):
    #                 for n in range(mat.shape[3]):
    #                     mato[m, :, :, n] = interpolate.interp2d(
    #                         x, y, mat[m, :, :, n], kind="cubic"
    #                     )(xnew, ynew)
    #             DATAUP.append(mato)

    #     return DATAUP
    
    def resample_data(self, DATA, k):
 
        # upsample by averaging
        if k > 1:
            kk = int(k)
            for im in range(len(DATA)):
                mat = DATA[im]
                t, m, n, o = mat.shape
                ny = m // kk
                nx = n // kk
                mato = np.nanmean( mat[:, : ny * kk, : nx * kk, :].reshape((t, ny, kk, nx, kk, o)), axis=(2, 4), )
                DATA[im] = mato 
                
        # downsample by cubic interpolation
        elif k < 1:
            for im in range(len(DATA)):
                mat = DATA[im]
                y = np.arange(mat.shape[1])
                x = np.arange(mat.shape[2])
                ynew = np.arange(int(mat.shape[1] / k)) * k
                ynew = ynew[ynew <= max(y)]
                xnew = np.arange(int(mat.shape[2] / k)) * k
                xnew = xnew[xnew <= max(x)]
                mato = np.zeros((mat.shape[0], len(ynew), len(xnew), mat.shape[3]))
                for m in range(mat.shape[0]):
                    for n in range(mat.shape[3]):
                        mato[m, :, :, n] = interpolate.interp2d( x, y, mat[m, :, :, n], kind="cubic" )(xnew, ynew)
                DATA[im] = mato 

    def type_of_file(self, pathlist):
        """
            Return the type of data file
        """

        pathncdf = [len(glob.glob(os.path.join(p, "ex.nc"))) > 0 for p in pathlist]
        pathnpy  = [len(glob.glob(os.path.join(p, "*.npy"))) > 0 for p in pathlist]

        if all(pathncdf):
            return "nc"
        elif all(pathnpy):
            return "npy"
        else:
            print("No data found with acceptable format (npy or ncdf)")
            sys.exit()

    def open_data(self):
        """
            Open data
        """

        print("----------------- OPEN  DATA        ------------------------")

        self.subdatasetname_train, self.subdatasetpath_train = self.findsubdata(
            os.path.join(self.config.data_dir, self.config.dataset, "train")
        )

        self.subdatasetname_test, self.subdatasetpath_test = self.findsubdata(
            os.path.join(self.config.data_dir, self.config.dataset, "test")
        )

        if self.config.include_test_in_train:
            self.subdatasetpath_train += self.subdatasetpath_test
            self.subdatasetname_train += self.subdatasetname_test

        if self.config.train:
            typefile = self.type_of_file(self.subdatasetpath_train)
            print('Format of the file found : ',typefile)
            print("----------------- TRAINIG DATA") 
            self.DATAIN_TRAIN, self.DATAOUT_TRAIN = self.open_dataset(
                self.subdatasetpath_train, typefile
            )

            if not self.config.resample_data == 1:
                print("----------------- RESAMPLING")                        
                self.resample_data(
                    self.DATAIN_TRAIN, self.config.resample_data
                )
                self.resample_data(
                    self.DATAOUT_TRAIN, self.config.resample_data
                )

        typefile = self.type_of_file(self.subdatasetpath_test)
        print("----------------- TEST DATA") 
        self.DATAIN_TEST, self.DATAOUT_TEST = self.open_dataset(
            self.subdatasetpath_test, typefile
        )

        if not self.config.resample_data == 1:
            self.resample_data(
                self.DATAIN_TEST, self.config.resample_data
            )
            self.resample_data(
                self.DATAOUT_TEST, self.config.resample_data
            )

        if self.config.patch_size < 0:
            tary = min([d.shape[1] for d in self.DATAIN_TEST + self.DATAIN_TRAIN])
            tarx = min([d.shape[2] for d in self.DATAIN_TEST + self.DATAIN_TRAIN])
            self.config.patch_size = 2 ** min(
                int(math.log(tarx) / math.log(2)), int(math.log(tary) / math.log(2))
            )
            print("Patch size auto-computed and equal to ", self.config.patch_size)

    def open_data_incremental(self):
        """
            Open data (for incremental training)
        """

        self.subdatasetname, self.subdatasetpath = self.findsubdata(
            os.path.join(self.config.data_dir, self.config.dataset)
        )

        typefile = self.type_of_file(self.subdatasetpath)
        self.DATAIN, self.DATAOUT = self.open_dataset(self.subdatasetpath, typefile)

        if not self.config.resample_data == 1:
            self.resample_data(self.DATAIN, self.config.resample_data)
            self.resample_data(self.DATAOUT, self.config.resample_data)
 
    ####################################################################################
    #                             ANALYZE DATA
    #################################################################################### 

    def analyze_data(self, DATAIN, DATAOUT, thr=0):
        """
            Routine to be used for a priori analysis of the data, especially to investigate the scaling
        """

        print("----------------- ANALYZE THE  DATA    ------------------------")

        DATAFLAT = []
        for k, j in enumerate(self.fieldin):
            DATAFLAT.append(
                np.concatenate(
                    [np.ndarray.flatten(datain[:, :, :, k]) for datain in DATAIN]
                )
            )

        for k, j in enumerate(self.fieldout):
            DATAFLAT.append(
                np.concatenate(
                    [np.ndarray.flatten(dataout[:, :, :, k]) for dataout in DATAOUT]
                )
            )

        if (self.config.maptype[0] == "f"):
            KEEP = DATAFLAT[0] > self.config.thrice
            for k, j in enumerate(self.fieldin + self.fieldout):
                DATAFLAT[k] = DATAFLAT[k][KEEP]

        from matplotlib.ticker import PercentFormatter

        os.makedirs(os.path.join(self.pathofresults, "hist_data"), exist_ok=True)

        for k, j in enumerate(self.fieldin + self.fieldout):

            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
            ax1.hist(DATAFLAT[k], 20, density=True, facecolor="g", alpha=0.75)
            ax1.yaxis.set_major_formatter(PercentFormatter(1))
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.pathofresults, "hist_data/hist-" + j + ".png"),
                pad_inches=0,
                dpi=100,
            )
            plt.close(fig)

        n0 = 0
        n1 = len(self.fieldin)
        n2 = len(self.fieldin) + len(self.fieldout)

        for k, j in enumerate(range(n0, n1)):

            if thr <= 0:
                a = np.min(DATAFLAT[j])
                b = np.max(DATAFLAT[j])
            else:
                a = np.quantile(DATAFLAT[j], thr)
                a = min(a, 0)
                b = np.quantile(DATAFLAT[j], 1 - thr)
                b = max(b, 0)

            print(self.fieldin[k], max(abs(a), abs(b)))

        for k, j in enumerate(range(n1, n2)):

            if thr <= 0:
                a = np.min(DATAFLAT[j])
                b = np.max(DATAFLAT[j])
            else:
                a = np.quantile(DATAFLAT[j], thr)
                a = min(a, 0)
                b = np.quantile(DATAFLAT[j], 1 - thr)
                b = max(b, 0)

            print(self.fieldout[k], max(abs(a), abs(b)))

    ####################################################################################
    #                           PATCH AND DATA GENERARTOR
    ####################################################################################

    def patch_generator(self, DATA):
        """
            Routine to serves to generate patches (NX x NY) tfor training
        """

        np.random.seed(self.config.seed)

        px = self.config.patch_size
        py = self.config.patch_size

        num_total = 0
        for data in DATA:
            num_total += (
                data.shape[0]
                * (int(data.shape[1] / py) + 1)
                * (int(data.shape[2] / px) + 1)
            )

        lkji = []

        for l, data in enumerate(DATA):

            num_samples = (
                5
                * data.shape[0]
                * (int(data.shape[1] / py) + 1)
                * (int(data.shape[2] / px) + 1)
            )

            F = []

            while len(F) < num_samples:

                k = np.random.randint(0, data.shape[0], num_samples)
                j = np.random.randint(0, data.shape[1] - py, num_samples)
                i = np.random.randint(0, data.shape[2] - px, num_samples)

                # this exclude patches with zero ice as not relevant for training
                if (self.config.maptype[0] == "f"):
                    F += [
                        [l, kk, jj, ii]
                        for (kk, jj, ii) in zip(k, j, i)
                        if data[kk, jj + int(py / 2), ii + int(px / 2), self.fieldin.index('thk')]
                        > self.config.thrice
                    ]
                else:
                    F += [[l, kk, jj, ii] for (kk, jj, ii) in zip(k, j, i)]

            lkji += F

        np.random.shuffle(lkji)

        return lkji, num_total

    # the 5 next function serves to data augmentation
    def applyrotations(self, M, id):
        if id == 1:
            M = np.rot90(M)
        if id == 2:
            M = np.rot90(np.rot90(M))
        if id == 3:
            M = np.rot90(np.rot90(np.rot90(M)))

    def applyflipud(self, M, id):
        if id == 1:
            M = np.flipud(M)

    def applyfliplr(self, M, id):
        if id == 1:
            M = np.fliplr(M)

    def applytranspose(self, M, id):
        if id == 1:
            M = np.transpose(M)

    def applyatransformation(self, M, id):
        self.applyrotations(M, id[0])
        self.applyflipud(M, id[1])
        self.applyfliplr(M, id[2])
        self.applytranspose(M, id[3])
        return M

    def datagenerator(self, DATAIN, DATAOUT, lkji):
        """
            Routine to serves to build a data generator
        """

        px = self.config.patch_size
        py = self.config.patch_size
        batch_size = self.config.batch_size

        while True:  # Loop forever so the generator never terminates

            for o in range(0, (len(lkji) // batch_size) * batch_size, batch_size):

                X = np.empty((batch_size, py, px, len(self.fieldin)))
                Y = np.empty((batch_size, py, px, len(self.fieldout)))

                for it in range(0, batch_size):

                    l, k, j, i = lkji[o + it]
                    if self.config.data_augmentation:
                        aug = [
                            np.random.randint(0, 4),
                            np.random.randint(0, 2),
                            np.random.randint(0, 2),
                            np.random.randint(0, 2),
                        ]
                    else:
                        aug = [0, 0, 0, 0]

                    for f in range(len(self.fieldin)):
                        X[it, :, :, f] = self.applyatransformation(
                            DATAIN[l][k, j : j + py, i : i + px, f]
                            / self.fieldbounds[self.fieldin[f]],
                            aug,
                        )

                    for f in range(len(self.fieldout)):
                        Y[it, :, :, f] = self.applyatransformation(
                            DATAOUT[l][k, j : j + py, i : i + px, f]
                            / self.fieldbounds[self.fieldout[f]],
                            aug,
                        )

                yield X, Y

    def create_data_generator(self):

        print(
            " ------ CREATE DATA GENERATOR TO FEED THE SOLVER -------------- "
        )

        if self.config.train:
            self.lkji_train, self.num_samples_train = self.patch_generator(
                self.DATAIN_TRAIN
            )
        self.lkji_test, self.num_samples_test = self.patch_generator(self.DATAIN_TEST)

        if self.config.train:
            self.trainSet = self.datagenerator(
                self.DATAIN_TRAIN, self.DATAOUT_TRAIN, self.lkji_train
            )
        self.testSet = self.datagenerator(
            self.DATAIN_TEST, self.DATAOUT_TEST, self.lkji_test
        )
 
    #################################################################################
    ######                      DEFINE CNN OR UNET                        ###########
    ################################################################################# 

    def cnn(self, nb_inputs, nb_outputs):
        """
            Routine serve to build a convolutional neural network
        """

        inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

        conv = inputs

        if self.config.activation == "lrelu":
            activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        else:
            activation = tf.keras.layers.ReLU()

        for i in range(int(self.config.nb_layers)):

            conv = tf.keras.layers.Conv2D(
                filters=self.config.nb_out_filter,
                kernel_size=(self.config.conv_ker_size, self.config.conv_ker_size),
                kernel_regularizer=tf.keras.regularizers.l2(self.config.regularization),
                padding="same",
            )(conv)

            conv = activation(conv)

            conv = tf.keras.layers.Dropout(self.config.dropout_rate)(conv)

        outputs = conv

        outputs = tf.keras.layers.Conv2D(filters=nb_outputs, \
                                         kernel_size=(1, 1), \
                                         activation=None)(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def unet(self, nb_inputs, nb_outputs):
        """
            Routine serve to define a UNET network from keras_unet_collection
        """

        from keras_unet_collection import models

        layers = np.arange(int(self.config.nb_blocks))

        number_of_filters = [
            self.config.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
        ]

        return models.unet_2d(
            (None, None, nb_inputs),
            number_of_filters,
            n_labels=nb_outputs,
            stack_num_down=2,
            stack_num_up=2,
            activation="LeakyReLU",
            output_activation=None,
            batch_norm=False,
            pool="max",
            unpool=False,
            name="unet",
        )
 
    #################################################################################
    ######                      TRAINING ROUTINE                          ###########
    ################################################################################# 

    def train(self):
        """
            THIS is the training routine
        """

        if self.config.train:

            print("=========== Define the type of network (CNN or sth else) =========== ")

            nb_inputs = len(self.fieldin)
            nb_outputs = len(self.fieldout)

            self.model = getattr(self, self.config.network)(nb_inputs, nb_outputs)

            print("=========== Deinfe the optimizer =========== ")

            if self.config.clipnorm == 0.0:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate
                )
            else:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.config.learning_rate,
                    clipnorm=self.config.clipnorm,
                )

            self.model.compile(
                loss=self.config.loss, optimizer=optimizer, metrics=["mae", "mse"]
            )  #  metrics=['mae','mse'] weighted_metrics=['mae','mse']

            step_per_epoch_train = self.num_samples_train // self.config.batch_size
            step_per_epoch_test = self.num_samples_test // self.config.batch_size

            print("=========== step_per_epoch : ", step_per_epoch_train)

            csv_logger = tf.keras.callbacks.CSVLogger(
                os.path.join(self.pathofresults, "train-history.csv"),
                append=True,
                separator=" ",
            )

            model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.pathofresults + "/model.{epoch:05d}.h5",
                save_freq="epoch",
                period=self.config.save_model_each,
            )
             
            class TimeHistory(tf.keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.times = []
            
                def on_epoch_begin(self, epoch, logs={}):
                    self.epoch_time_start = time.time()
            
                def on_epoch_end(self, epoch, logs={}):
                    self.times.append(time.time() - self.epoch_time_start)

            time_cb = TimeHistory()

            TerminateOnNaN_cb = tf.keras.callbacks.TerminateOnNaN()

            EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                min_delta=0,
                patience=100,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )

            cb = [
                csv_logger,
                time_cb,
                TerminateOnNaN_cb,
                EarlyStopping_cb,
                model_checkpoint_cb,
            ]

            print(self.model.summary())

            original_stdout = (
                sys.stdout
            )  # Save a reference to the original standard output
            with open(os.path.join(self.pathofresults, "model-info.txt"), "w") as f:
                sys.stdout = f  # Change the standard output to the file we created.
                print(self.model.summary())
                sys.stdout = (
                    original_stdout  # Reset the standard output to its original value
                )

            print("=========== TRAINING =========== ")
            history = self.model.fit(
                self.trainSet,
                validation_data=self.testSet,
                # validation_freq=10,
                validation_steps=step_per_epoch_test,
                batch_size=self.config.batch_size,
                steps_per_epoch=step_per_epoch_train,
                epochs=self.config.epochs,
                callbacks=cb,
                verbose=self.config.verbose,
            )

            print("=========== plot learning curves =========== ")
            self.plotlearningcurves(self.pathofresults, history.history)

            print("=========== save information on computational times =========== ")
            with open(
                os.path.join(self.pathofresults, "train-time_callback.txt"), "w"
            ) as ff:
                print(time_cb.times, file=ff)

            self.history = history.history
            self.model.save(self.modelfile)

        else:
            self.model = tf.keras.models.load_model(self.modelfile)
            
    def plotlearningcurves(self, pathofresults, hist):

        fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hist["loss"], "-", label="Train loss")
        ax.plot(hist["val_loss"], "--", label="Validation loss")
        ax.set_xlabel("Epoch", size=15)
        ax.set_ylabel("Loss", size=15)
        ax.legend(fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(pathofresults, "Learning-curve.pdf"))
        plt.close("all")

    #################################################################################
    ######                      PREDICT / EVALUATION ROUTINES             ###########
    ################################################################################# 

    def predict(self):
        """
            THIS is the predict routine for validation after training
        """

        if self.config.predict:

            print("===========     PREDICTING STEP =========== ")

            for kk, predglacier in enumerate(self.subdatasetname_test):

                DATAIN_T = [self.DATAIN_TEST[kk]]
                DATAOUT_T = [self.DATAOUT_TEST[kk]]

                DATAOUT_P, eval_time = self.evaluate(DATAIN_T)

                pathofresults = os.path.join(self.pathofresults, predglacier)

                os.makedirs(pathofresults, exist_ok=True)

                self.plotresu(pathofresults, predglacier, DATAOUT_T, DATAOUT_P)

                if self.config.maptype[0] == "f":
                    self.additional_post_processing(DATAIN_T,DATAOUT_T,DATAOUT_P,eval_time,pathofresults)
                    
    def additional_post_processing(self,DATAIN_T,DATAOUT_T,DATAOUT_P,eval_time,pathofresults):
        
        print('no additional processing here')        

    def evaluate(self, DATAIN):
        """
            THIS function evaluates the neural network 
        """

        PRED = []

        for datain in DATAIN:

            Nt = datain.shape[0]
            Ny = datain.shape[1]
            Nx = datain.shape[2]

            if self.config.network == "unet":
                multiple_window_size = 8  # maybe this 2**(nb_layers-1)
                NNy = multiple_window_size * math.ceil(Ny / multiple_window_size)
                NNx = multiple_window_size * math.ceil(Nx / multiple_window_size)
            else:
                NNy = Ny
                NNx = Nx

            PAD = [[0, NNy - Ny], [0, NNx - Nx]]

            PREDI = np.zeros((Nt, Ny, Nx, len(self.fieldout)))

            eval_time = []

            for k in range(Nt):

                start_time = time.time()

                X = np.expand_dims(
                    np.stack(
                        [
                            np.pad(datain[k, :, :, kk], PAD) / self.fieldbounds[f]
                            for kk, f in enumerate(self.fieldin)
                        ],
                        axis=-1,
                    ),
                    axis=0,
                )

                Y = self.model.predict_on_batch(X)

                PREDI[k, :, :, :] = np.stack(
                    [
                        Y[0, :Ny, :Nx, kk] * self.fieldbounds[f]
                        for kk, f in enumerate(self.fieldout)
                    ],
                    axis=-1,
                )

                eval_time.append(time.time() - start_time)

            PRED.append(PREDI)

        return PRED, np.mean(eval_time)

    ####################################################################################
    #                             PLOT RESULT FROM PREDICT
    ####################################################################################

    def plotresu(self, pathofresults, dataset, DATAOUT_T, DATAOUT_P):

        for l in range(len(DATAOUT_T)):
            step = max(int(len(DATAOUT_T[l]) / 10), 1)
            for k in range(0, len(DATAOUT_T[l]), step):
                print("Plotting snapshopt n° : ", k)
                self.plotresu_f(pathofresults, dataset, DATAOUT_P[l][k], DATAOUT_T[l][k], k)

    def plotresu_f(self, pathofresults, dataset, pred_outputs, true_outputs, idx):
        """
            This routine permtis to plot predicted output against validation output
        """
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib import cm

        ny = pred_outputs.shape[0]
        nx = pred_outputs.shape[1]
        my_dpi = 100

        fig, (ax2, ax1) = plt.subplots(
            1, 2, figsize=(2 * ny / my_dpi, nx / my_dpi), dpi=my_dpi
        )

        m1 = tf.keras.metrics.MeanAbsoluteError()
        mae = m1(pred_outputs, true_outputs)

        m2 = tf.keras.metrics.RootMeanSquaredError()
        mse = m2(pred_outputs, true_outputs)

        pred_outputs0 = np.linalg.norm(pred_outputs, axis=2)
        true_outputs0 = np.linalg.norm(true_outputs, axis=2)

        valmaxx = max(
            [np.quantile(pred_outputs0, 0.999), np.quantile(true_outputs0, 0.999)]
        )

        ############################################

        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.set_title("PREDICTED, mae: %.5f, mse: %.5f" % (mae, mse))
        im1 = ax1.imshow(
            pred_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")

        ax2.set_title("TRUE, Id: %s" % str(idx))
        im2 = ax2.imshow(
            true_outputs0,
            origin="lower",
            vmin=0,
            vmax=valmaxx,
            cmap=cm.get_cmap("viridis", 10),
        )
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(im2, format="%.2f", cax=cax2)
        ax2.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                pathofresults, "predict-" + dataset + "_" + str(idx).zfill(4) + ".png"
            ),
            pad_inches=0,
            dpi=100,
        )
        plt.close("all")

        ############################################

        tod = pred_outputs0 - true_outputs0

        tod[np.abs(tod) < 0.03] = np.nan

        valmaxx2 = 25  # np.quantile(tod,0.99)

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

        ax1.set_title("PREDICTED - TRUE")
        im1 = ax1.imshow(
            tod,
            origin="lower",
            vmin=-valmaxx2,
            vmax=valmaxx2,
            cmap=cm.get_cmap("RdBu", 10),
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, format="%.2f", cax=cax1)
        ax1.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                pathofresults,
                "predict-diff_" + dataset + "_" + str(idx).zfill(4) + ".png",
            ),
            pad_inches=0,
            dpi=100,
        )
        plt.close("all")

####################################################################################
#                               RUN
####################################################################################

    def run(self):
        """
            This is the main routine, which list the workflow
        """
        
        self.initialize()
        
        self.open_data()
        
        self.create_data_generator()
        
        self.train()
        
        self.predict()

####################################################################################
#                               END
####################################################################################
