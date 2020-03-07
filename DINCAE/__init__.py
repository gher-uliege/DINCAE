#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DINCAE: Data-Interpolating Convolutional Auto-Encoder
# Copyright (C) 2019 Alexander Barth
#
# This file is part of DINCAE.

# DINCAE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# DINCAE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DINCAE. If not, see <http://www.gnu.org/licenses/>.

"""
DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations.

For most application it is sufficient to call the function
`DINCAE.reconstruct_gridded_nc` directly.

The code is available at:
[https://github.com/gher-ulg/DINCAE](https://github.com/gher-ulg/DINCAE)
"""
import os
import random
from math import ceil, floor
from netCDF4 import Dataset, num2date
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


__all__ = ["reconstruct","load_gridded_nc","data_generator","reconstruct_gridded_nc"]


def identity(x):
    return x

def load_gridded_nc(fname,varname, minfrac = 0.05):
    """
Load the variable `varname` from the NetCDF file `fname`. The variable `lon` is
the longitude in degrees east, `lat` is the latitude in degrees North, `time` is
a numpy datetime vector, `data_full` is a 3-d array with the data, `missing`
is a boolean mask where true means the data is missing and `mask` is a boolean mask
where true means the data location is valid, e.g. sea points for sea surface temperature.

At the bare-minimum a NetCDF file should have the following variables and
attributes:


    netcdf file.nc {
    dimensions:
            time = UNLIMITED ; // (5266 currently)
            lat = 112 ;
            lon = 112 ;
    variables:
            double lon(lon) ;
            double lat(lat) ;
            double time(time) ;
                    time:units = "days since 1900-01-01 00:00:00" ;
            int mask(lat, lon) ;
            float SST(time, lat, lon) ;
                    SST:_FillValue = -9999.f ;
    }

"""
    ds = Dataset(fname);
    lon = ds.variables["lon"][:].data;
    lat = ds.variables["lat"][:].data;
    time = num2date(ds.variables["time"][:],ds.variables["time"].units);

    data = ds.variables[varname][:,:,:];

    if "mask" in ds.variables:
        mask = ds.variables["mask"][:,:].data == 1;
    else:
        print("compute mask for ",varname,": sea point should have at least ",
              minfrac," for valid data tought time")

        if np.isscalar(data.mask):
            mask = np.ones((data.shape[1],data.shape[2]),dtype=np.bool)
        else:
            mask = np.mean(~data.mask,axis=0) > minfrac


        print("mask: sea points ",np.sum(mask))
        print("mask: land points ",np.sum(~mask))

    print("varname ",varname,mask.shape)
    ds.close()

    if np.isscalar(data.mask):
        missing = np.zeros(data.shape,dtype=np.bool)
    else:
        missing = data.mask

    print("data shape, range",data.shape,data.min(),data.max())
    return lon,lat,time,data,missing,mask


def data_generator(lon,lat,time,data_full,missing,
                   train = True,
                   ntime_win = 3,
                   obs_err_std = 1.,
                   jitter_std = 0.05):

    return data_generator_list(lon,lat,time,[data_full],[missing],
                   train = True,
                   ntime_win = ntime_win,
                   obs_err_std = [obs_err_std],
                   jitter_std = [jitter_std])

def data_generator_list(lon,lat,time,data_full,missing,
                   train = True,
                   ntime_win = 3,
                   obs_err_std = [1.],
                   jitter_std = [0.05]):
    """
Return a generator for training (`train = True`) or testing (`train = False`)
the neural network. `obs_err_std` is the error standard deviation of the
observations. The variable `lon` is the longitude in degrees east, `lat` is the
latitude in degrees North, `time` is a numpy datetime vector, `data_full` is a
3-d array with the data and `missing` is a boolean mask where true means the data is
missing. `jitter_std` is the standard deviation of the noise to be added to the
data during training.

The output of this function is `datagen`, `ntime` and `meandata`. `datagen` is a
generator function returning a single image (relative to the mean `meandata`),
`ntime` the number of time instances for training or testing and `meandata` is
the temporal mean of the data.

    # number of time instances, must be odd
    ntime_win = 3

"""
    sz = data_full[0].shape
    print("sz ",sz)
    ntime = sz[0]
    ndata = len(data_full)

    dayofyear = np.array([d.timetuple().tm_yday for d in time])
    dayofyear_cos = np.cos(dayofyear/365.25)
    dayofyear_sin = np.sin(dayofyear/365.25)

    meandata = [None] * ndata
    data = [None] * ndata

    for i in range(ndata):
        meandata[i] = data_full[i].mean(axis=0,keepdims=True)
        data[i] = data_full[i] - meandata[i]

        if data_full[i].shape != data_full[0].shape:
            raise ArgumentError("shape are not coherent")


    # scaled mean and inverse of error variance for every input data
    # plus lon, lat, cos(time) and sin(time)
    x = np.zeros((sz[0],sz[1],sz[2],2*ndata + 4),dtype="float32")

    for i in range(ndata):
        x[:,:,:,2*i] = data[i].filled(0) / (obs_err_std[i]**2)
        x[:,:,:,2*i+1] = (1-data[i].mask) / (obs_err_std[i]**2)  # error variance

    # scale between -1 and 1
    lon_scaled = 2 * (lon - np.min(lon)) / (np.max(lon) - np.min(lon)) - 1
    lat_scaled = 2 * (lat - np.min(lat)) / (np.max(lat) - np.min(lat)) - 1

    i = 2*ndata
    x[:,:,:,i  ] = lon_scaled.reshape(1,1,len(lon))
    x[:,:,:,i+1] = lat_scaled.reshape(1,len(lat),1)
    x[:,:,:,i+2] = dayofyear_cos.reshape(len(dayofyear_cos),1,1)
    x[:,:,:,i+3] = dayofyear_sin.reshape(len(dayofyear_sin),1,1)

    nvar = 2 * ntime_win * ndata + 4

    # generator for data
    def datagen():
        for i in range(ntime):
            xin = np.zeros((sz[1],sz[2],nvar),dtype="float32")
            xin[:,:,0:(2*ndata + 4)]  = x[i,:,:,:]

            ioffset = (2*ndata + 4)
            for time_index in range(0,ntime_win):
                # nn is centered on the current time, e.g. -1 (past), 0 (present), 1 (future)
                nn = time_index - (ntime_win//2)
                # current time is already included, skip it
                if nn != 0:
                    i_clamped = min(ntime-1,max(0,i+nn))
                    xin[:,:,ioffset:(ioffset + 2*ndata)] = x[i_clamped,:,:,0:(2*ndata)]
                    ioffset = ioffset + 2*ndata

            # add missing data during training randomly
            if train:
                #imask = random.randrange(0,missing.shape[0])
                imask = random.randrange(0,ntime)

                for j in range(ndata):
                    selmask = missing[j][imask,:,:]
                    xin[:,:,2*j][selmask] = 0
                    xin[:,:,2*j+1][selmask] = 0

                # add jitter
                for j in range(ndata):
                    xin[:,:,2*j] += jitter_std[j] * np.random.randn(sz[1],sz[2])
                    xin[:,:,2*j + 2*ndata + 4] += jitter_std[j] * np.random.randn(sz[1],sz[2])
                    xin[:,:,2*j + 4*ndata + 4] += jitter_std[j] * np.random.randn(sz[1],sz[2])

            yield (xin,x[i,:,:,0:2])

    # meandata[0] is the primary variable to be reconstructed
    return datagen,nvar,ntime,meandata[0]



def savesample(fname,m_rec,σ2_rec,meandata,lon,lat,e,ii,offset,
               transfun = (identity, identity)):
    fill_value = -9999.
    recdata = transfun[1](m_rec  + meandata)
    # todo apply transfun to sigma_rec

    if transfun[1] == np.exp:
        # relative error
        #sigma_rec = recdata * np.sqrt(σ2_rec)
        sigma_rec = np.sqrt(σ2_rec) # debug
    elif transfun[1] == identity:
        sigma_rec = np.sqrt(σ2_rec)
    else:
        print("warning: sigma_rec is not transformed")
        sigma_rec = np.sqrt(σ2_rec)


    if ii == 0:
        # create file
        root_grp = Dataset(fname, 'w', format='NETCDF4')

        # dimensions
        root_grp.createDimension('time', None)
        root_grp.createDimension('lon', len(lon))
        root_grp.createDimension('lat', len(lat))

        # variables
        nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
        nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
        nc_meandata = root_grp.createVariable(
            'meandata', 'f4', ('lat','lon'),
            fill_value=fill_value)

        nc_mean_rec = root_grp.createVariable(
            'mean_rec', 'f4', ('time', 'lat', 'lon'),
            fill_value=fill_value)

        nc_sigma_rec = root_grp.createVariable(
            'sigma_rec', 'f4', ('time', 'lat', 'lon',),
            fill_value=fill_value)

        # data
        nc_lon[:] = lon
        nc_lat[:] = lat
        nc_meandata[:,:] = meandata
    else:
        # append to file
        root_grp = Dataset(fname, 'a')
        nc_mean_rec = root_grp.variables['mean_rec']
        nc_sigma_rec = root_grp.variables['sigma_rec']

    for n in range(m_rec.shape[0]):
#        nc_mean_rec[n+offset,:,:] = np.ma.masked_array(
#            recdata[n,:,:],meandata.mask)
        nc_mean_rec[n+offset,:,:] = np.ma.masked_array(
            recdata[n,:,:],meandata.mask)
        nc_sigma_rec[n+offset,:,:] = np.ma.masked_array(
            sigma_rec[n,:,:],meandata.mask)


    root_grp.close()


# save inversion
def sinv(x, minx = 1e-3):
    return 1 / tf.maximum(x,minx)


def reconstruct(lon,lat,mask,meandata,
                train_datagen,train_len,
                test_datagen,test_len,
                outdir,
                resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                epochs = 1000,
                batch_size = 30,
                save_each = 10,
                save_model_each = 500,
                skipconnections = [1,2,3,4],
                dropout_rate_train = 0.3,
                tensorboard = False,
                truth_uncertain = False,
                shuffle_buffer_size = 3*15,
                nvar = 10,
                enc_ksize_internal = [16,24,36,54],
                frac_dense_layer = [0.2],
                clip_grad = 5.0,
                regularization_L2_beta = 0,
                transfun = (identity,identity),
                savesample = savesample,
                learning_rate = 1e-3,
                learning_rate_decay_epoch = 100,
                iseed = None,
                nprefetch = 0,
                loss = [],
                nepoch_keep_missing = 0,
):
    """
Train a neural network to reconstruct missing data using the training data set
and periodically run the neural network on the test dataset. The function returns the
filename of the latest reconstruction.

## Parameters

 * `lon`: longitude in degrees East
 * `lat`: latitude in degrees North
 * `mask`:  boolean mask where true means the data location is valid,
e.g. sea points for sea surface temperature.
 * `meandata`: the temporal mean of the data.
 * `train_datagen`: generator function returning a single image for training
 * `train_len`: number of training images
 * `test_datagen`: generator function returning a single image for testing
 * `test_len`: number of testing images
 * `outdir`: output directory

## Optional input arguments

 * `resize_method`: one of the resize methods defined in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)
 * `epochs`: number of epochs for training the neural network
 * `batch_size`: size of a mini-batch
 * `save_each`: reconstruct the missing data every `save_each` epoch. Repeated saving is disabled if `save_each` is zero. The last epoch is always saved.
 * `save_model_each`: save a checkpoint of the neural network every
      `save_model_each` epoch
 * `skipconnections`: list of indices of convolutional layers with
     skip-connections
 * `dropout_rate_train`: probability for drop-out during training
 * `tensorboard`: activate tensorboard diagnostics
 * `truth_uncertain`: how certain you are about the perceived truth?
 * `shuffle_buffer_size`: number of images for the shuffle buffer
 * `nvar`: number of input variables
 * `enc_ksize_internal`: kernel sizes for the internal convolutional layers
      (after the input convolutional layer)
 * `clip_grad`: clip gradient to a maximum L2-norm.
 * `regularization_L2_beta`: scalar to enforce L2 regularization on the weight
 * `learning_rate`:  The initial learning rate
 * `learning_rate_decay_epoch`: The exponential recay rate of the leaning rate. After `learning_rate_decay_epoch` the learning rate is halved. The learning rate is compute as  `learning_rate * 0.5^(epoch / learning_rate_decay_epoch)`. `learning_rate_decay_epoch` can be `numpy.inf` for a constant learning rate
"""


    if iseed != None:
        np.random.seed(iseed)
        tf.compat.v1.set_random_seed(np.random.randint(0,2**32-1))
        random.seed(np.random.randint(0,2**32-1))

    print("regularization_L2_beta ",regularization_L2_beta)
    print("enc_ksize_internal ",enc_ksize_internal)
    print("nvar ",nvar)
    print("nepoch_keep_missing ",nepoch_keep_missing)

    enc_ksize = [nvar] + enc_ksize_internal

    if outdir != None:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

    jmax,imax = mask.shape

    sess = tf.compat.v1.Session()

    # Repeat the input indefinitely.
    # training dataset iterator
    train_dataset = tf.data.Dataset.from_generator(
        train_datagen, (tf.float32,tf.float32),
        (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).repeat().shuffle(shuffle_buffer_size).batch(batch_size)

    train_iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
    train_iterator_handle = sess.run(train_iterator.string_handle())

    # test dataset without added clouds
    # must be reinitializable
    test_dataset = tf.data.Dataset.from_generator(
        test_datagen, (tf.float32,tf.float32),
        (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).batch(batch_size)

    if nprefetch > 0:
        train_dataset = train_dataset.prefetch(nprefetch)
        test_dataset = test_dataset.prefetch(nprefetch)

    test_iterator = tf.compat.v1.data.Iterator.from_structure(test_dataset.output_types,
                                                    test_dataset.output_shapes)
    test_iterator_init_op = test_iterator.make_initializer(test_dataset)

    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[], name = "handle_name_iterator")
    iterator = tf.compat.v1.data.Iterator.from_string_handle(
            handle, train_iterator.output_types, output_shapes = train_iterator.output_shapes)


    inputs_,xtrue = iterator.get_next()



    # Encoder

    enc_nlayers = len(enc_ksize)
    enc_conv = [None] * enc_nlayers
    enc_avgpool = [None] * enc_nlayers

    enc_avgpool[0] = inputs_

    for l in range(1,enc_nlayers):
        enc_conv[l] = tf.compat.v1.layers.conv2d(enc_avgpool[l-1],
                                       enc_ksize[l],
                                       (3,3),
                                       padding='same',
                                       activation=tf.nn.leaky_relu)
        print("encoder: output size of convolutional layer: ",l,enc_conv[l].shape)

        enc_avgpool[l] = tf.compat.v1.layers.average_pooling2d(enc_conv[l],
                                                     (2,2),
                                                     (2,2),
                                                     padding='same')

        print("encoder: output size of pooling layer: ",l,enc_avgpool[l].shape)

        enc_last = enc_avgpool[-1]

    # default is no drop-out
    dropout_rate = tf.compat.v1.placeholder_with_default(0.0, shape=())

    if len(frac_dense_layer) == 0:
        dense_2d = enc_last
    else:
        # Dense Layers
        ndensein = enc_last.shape[1:].num_elements()
        print("ndensein ",ndensein)

        avgpool_flat = tf.reshape(enc_last, [-1, ndensein])

        # number of output units for the dense layers
        dense_units = [floor(ndensein*frac) for frac in frac_dense_layer + list(reversed(frac_dense_layer[:-1]))]
        # last dense layer must give again the same number as input units
        dense_units.append(ndensein)

        dense = [None] * (4*len(frac_dense_layer)+1)
        dense[0] = avgpool_flat

        for i in range(2*len(frac_dense_layer)):
            dense[2*i+1] = tf.compat.v1.layers.dense(inputs=dense[2*i],
                                           units=dense_units[i],
                                           activation=tf.nn.relu)
            print("dense layer: output units: ",i,dense[2*i+1].shape)
            dense[2*i+2] = tf.compat.v1.layers.dropout(inputs=dense[2*i+1], rate=dropout_rate)

        dense_2d = tf.reshape(dense[-1], tf.shape(input=enc_last))

    ### Decoder
    dec_conv = [None] * enc_nlayers
    dec_upsample = [None] * enc_nlayers

    dec_conv[0] = dense_2d

    for l in range(1,enc_nlayers):
        l2 = enc_nlayers-l

        dec_upsample[l] = tf.image.resize(
            dec_conv[l-1],
            enc_conv[l2].shape[1:3],
            method=resize_method)
        print("decoder: output size of upsample layer: ",l,dec_upsample[l].shape)

        # short-cut
        if l in skipconnections:
            print("skip connection at ",l)
            dec_upsample[l] = tf.concat([dec_upsample[l],enc_avgpool[l2-1]],3)
            print("decoder: output size of concatenation: ",l,dec_upsample[l].shape)

        dec_conv[l] = tf.compat.v1.layers.conv2d(
            dec_upsample[l],
            enc_ksize[l2-1],
            (3,3),
            padding='same',
            activation=tf.nn.leaky_relu)

        print("decoder: output size of convolutional layer: ",l,dec_conv[l].shape)

    # last layer of decoder
    xrec = dec_conv[-1]

    loginvσ2_rec = xrec[:,:,:,1]
    invσ2_rec = tf.exp(tf.minimum(loginvσ2_rec,10))
    σ2_rec = sinv(invσ2_rec)
    m_rec = xrec[:,:,:,0] * σ2_rec


    σ2_true = sinv(xtrue[:,:,:,1])
    m_true = xtrue[:,:,:,0] * σ2_true
    σ2_in = sinv(inputs_[:,:,:,1])
    m_in = inputs_[:,:,:,0] * σ2_in


    difference = m_rec - m_true

    mask_issea = tf.compat.v1.placeholder(
        tf.float32,
        shape = (mask.shape[0], mask.shape[1]),
        name = "mask_issea")

    # 1 if measurement
    # 0 if no measurement (cloud or land for SST)
    mask_noncloud = tf.cast(tf.math.logical_not(tf.equal(xtrue[:,:,:,1], 0)),
                            xtrue.dtype)

    n_noncloud = tf.reduce_sum(input_tensor=mask_noncloud)

    if truth_uncertain:
        # KL divergence between two univariate Gaussians p and q
        # p ~ N(σ2_1,\mu_1)
        # q ~ N(σ2_2,\mu_2)
        #
        # 2 KL(p,q) = log(σ2_2/σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
        # 2 KL(p,q) = log(σ2_2) - log(σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
        # 2 KL(p_true,q_rec) = log(σ2_rec/σ2_true) + (σ2_true + (\mu_rec - \mu_true)^2)/(σ2_rec) - 1

        cost = (tf.reduce_sum(input_tensor=tf.multiply(
            tf.math.log(σ2_rec/σ2_true) + (σ2_true + difference**2) / σ2_rec,mask_noncloud))) / n_noncloud
    else:
        cost = (tf.reduce_sum(input_tensor=tf.multiply(tf.math.log(σ2_rec),mask_noncloud)) +
            tf.reduce_sum(input_tensor=tf.multiply(difference**2 / σ2_rec,mask_noncloud))) / n_noncloud


    # L2 regularization of weights
    if regularization_L2_beta != 0:
        trainable_variables   = tf.compat.v1.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_variables
                            if 'bias' not in v.name ]) * regularization_L2_beta
        cost = cost + lossL2

    RMS = tf.sqrt(tf.reduce_sum(input_tensor=tf.multiply(difference**2,mask_noncloud))
                  / n_noncloud)

    # to debug
    # cost = RMS

    if tensorboard:
        with tf.compat.v1.name_scope('Validation'):
            tf.compat.v1.summary.scalar('RMS', RMS)
            tf.compat.v1.summary.scalar('cost', cost)
            tf.compat.v1.summary.image("m_rec",tf.expand_dims(
                tf.reverse(tf.multiply(m_rec,mask_issea),[1]),-1))
            tf.compat.v1.summary.image("m_true",tf.expand_dims(
                tf.reverse(tf.multiply(m_true,mask_issea),[1]),-1))
            tf.compat.v1.summary.image("sigma2_rec",tf.expand_dims(
                tf.reverse(tf.multiply(σ2_rec,mask_issea),[1]),-1))

    # parameters for Adam optimizer (default values)
    #learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08

    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = 1e-3
    # learning_rate = tf.train.exponential_decay(starter_learning_rate,
    #                                                      global_step,
    #                                                      50, 0.96, staircase=True)

    learning_rate_ = tf.compat.v1.placeholder(tf.float32, shape=[])

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_,beta1,beta2,epsilon)
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients, _ = tf.clip_by_global_norm(gradients, clip_grad)
    opt = optimizer.apply_gradients(zip(gradients, variables))

    # Passing global_step to minimize() will increment it at each step.
    # opt = (
    #     tf.train.GradientDescentOptimizer(learning_rate)
    #     .minimize(cost, global_step=global_step)
    # )

    # optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)


    dt_start = datetime.now()
    print(dt_start)

    if tensorboard:
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(outdir + '/train',
                                          sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(outdir + '/test')
    else:
        # unused
        merged = tf.constant(0.0, shape=[1], dtype="float32")

    index = 0

    print("init")
    sess.run(tf.compat.v1.global_variables_initializer())
    logger.debug('init done')

    saver = tf.compat.v1.train.Saver()

    # final output file name
    fname = None

    # loop over epochs
    for e in range(epochs):
        if nepoch_keep_missing > 0:
            # use same clouds for every e.g. 20 epochs
            random.seed(iseed + e//nepoch_keep_missing)


        # loop over training datasets
        for ii in range(ceil(train_len / batch_size)):

            # run a single step of the optimizer
            #logger.debug(f'running {ii}')
            summary, batch_cost, batch_RMS, bs, batch_learning_rate, _ = sess.run(
                [merged, cost, RMS, mask_noncloud, learning_rate_, opt],feed_dict={
                    handle: train_iterator_handle,
                    mask_issea: mask,
                    learning_rate_: learning_rate * (0.5 ** (e / learning_rate_decay_epoch)),
                    dropout_rate: dropout_rate_train})

            #logger.debug('running done')
            loss.append(batch_cost)

            if tensorboard:
                train_writer.add_summary(summary, index)

            index += 1

            if ii % 20 == 0:
            #if ii % 1 == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training loss: {:.20f}".format(batch_cost),
                      "RMS: {:.20f}".format(batch_RMS), batch_learning_rate )


        if ((e == epochs-1) or ((save_each > 0) and (e % save_each == 0))) and outdir != None:
            print("Save output",e)

            timestr = datetime.now().strftime("%Y-%m-%dT%H%M%S")
            fname = os.path.join(outdir,"data-{}.nc".format(timestr))

            # reset test iterator, so that we start from the beginning
            sess.run(test_iterator_init_op)

            for ii in range(ceil(test_len / batch_size)):
                summary, batch_cost,batch_RMS,batch_m_rec,batch_σ2_rec = sess.run(
                    [merged, cost,RMS,m_rec,σ2_rec],
                    feed_dict = { handle: test_iterator_handle,
                                  mask_issea: mask })

                # time instances already written
                offset = ii*batch_size
                savesample(fname,batch_m_rec,batch_σ2_rec,meandata,lon,lat,e,ii,
                           offset, transfun = transfun)

        if ((save_model_each > 0) and (e % save_model_each == 0)) and outdir != None:
            save_path = saver.save(sess, os.path.join(
                outdir,"model-{:03d}.ckpt".format(e+1)))

    # free all resources associated with the session
    sess.close()

    dt_end = datetime.now()
    print(dt_end)
    print(dt_end - dt_start)

    return fname

def reconstruct_gridded_nc(filename,varname,outdir,
                           jitter_std = 0.05,
                           ntime_win = 3,
                           transfun = (identity, identity),
                           **kwargs):
    """
Train a neural network to reconstruct missing data from the NetCDF variable
`varname` in the NetCDF file `filename`. Results are saved in the output
directory `outdir`. `jitter_std` is the standard deviation of the noise to be
added to the data during training.
See `DINCAE.reconstruct` for other keyword arguments and
`DINCAE.load_gridded_nc` for the NetCDF format.

"""

    lon,lat,time,data,missing,mask = load_gridded_nc(filename,varname)
    data_trans = transfun[0](data)

    train_datagen,nvar,train_len,meandata = data_generator(
        lon,lat,time,data,missing,
        ntime_win = ntime_win,
        jitter_std = jitter_std)
    test_datagen,nvar,test_len,test_meandata = data_generator(
        lon,lat,time,data,missing,
        ntime_win = ntime_win,
        train = False)

    print("Number of input variables: ",nvar)

    reconstruct(
        lon,lat,mask,meandata,
        train_datagen,train_len,
        test_datagen,test_len,
        outdir,
        transfun = transfun,
        nvar = nvar,
        **kwargs)



def reconstruct_gridded_files(fields,outdir,
                              ntime_win = 3,
                              **kwargs):
    """
Train a neural network to reconstruct missing data from the NetCDF variable
`varname` in the NetCDF file `filename`. Results are saved in the output
directory `outdir`. `jitter_std` is the standard deviation of the noise to be
added to the data during training.
See `DINCAE.reconstruct` for other keyword arguments and
`DINCAE.load_gridded_nc` for the NetCDF format.

"""

    data_full = [None] * len(fields)
    missing = [None] * len(fields)
    jitter_std = [None] * len(fields)
    transfun = [None] * len(fields)
    varnames = [None] * len(fields)
    obs_err_std = [1] * len(fields) # value is irrelevant
    lon = []
    lat = []
    time = []

    for (i,field) in enumerate(fields):
        transfun[i] = field.get("transfun",(identity,identity))
        varnames[i] = field["varname"]

        field["lon"],field["lat"],field["time"],field["data"],field["missing"],field["mask"] = load_gridded_nc(field["filename"],field["varname"])

        data_full[i] = transfun[i][0](field["data"])

        print("typeof- ",type(field["data"]))
        print("typeof ",type(data_full[i]))

        missing[i] = field["missing"]
        jitter_std[i] = field.get("jitter_std",0)

    lon = fields[0]["lon"]
    lat = fields[0]["lat"]
    time = fields[0]["time"]
    mask = fields[0]["mask"]

    ndata = len(fields)

    train_datagen,nvar,train_len,meandata = data_generator_list(
        lon,lat,time,data_full,missing,
        obs_err_std = obs_err_std,
        jitter_std = jitter_std,
        ntime_win = ntime_win,
    )
    test_datagen,nvar,test_len,test_meandata = data_generator_list(
        lon,lat,time,data_full,missing,
        obs_err_std = obs_err_std,
        ntime_win = ntime_win,
        train = False)

    print("Number of input variables: ",nvar)

    fname = reconstruct(
        lon,lat,mask,meandata,
        train_datagen,train_len,
        test_datagen,test_len,
        outdir,
        transfun = transfun[0],
        nvar = nvar,
        **kwargs)

    return fname

#  LocalWords:  DINCAE Convolutional MERCHANTABILITY gridded
#  LocalWords:  TensorBoard stddev varname NetCDF fname lon numpy datetime
#  LocalWords:  boolean netcdf FillValue obs_err_std datagen ntime meandata
