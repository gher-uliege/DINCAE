using DINCAE
using TensorFlow
const tf = TensorFlow
BILINEAR = 0
NEAREST_NEIGHBOR = 1
BICUBIC = 2
AREA = 3

fname = "avhrr_sub_add_clouds.nc"
varname = "SST"

lon,lat,datatime,data_full,missingmask,mask = DINCAE.load_gridded_nc(fname,varname)


#datagen,ntimes,meandata = DINCAE.data_generator(lon,lat,datatime,data_full,missingmask, train = false)
#xin,xtrue = datagen(i);
outdir = "/tmp/DINCAE.jl/"


                #resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR# ,
                resize_method = NEAREST_NEIGHBOR# ,
                epochs = 1000# ,
                batch_size = 30# ,
                save_each = 10# ,
                save_model_each = 500# ,
                skipconnections = [1,2,3,4]# ,
                dropout_rate_train = 0.3# ,
                tensorboard = false# ,
                truth_uncertain = false# ,
                shuffle_buffer_size = 3*15# ,
                nvar = 10# ,
                enc_ksize_internal = [16,24,36,54]# ,
                clip_grad = 5.0# ,
                regularization_L2_beta = 0



dd = DINCAE.NCData(lon,lat,datatime,data_full,missingmask)

i = 1;
xin,xtrue = dd[i];

batch_size = 2;

DINCAE.RVec(dd)

p = partition(DINCAE.RVec(dd),batch_size);
batch = first(p);

inputs_ = cat((b[1] for b in batch)... , dims = Val(4));
xtrue = cat((b[2] for b in batch)... , dims = Val(4));





    jmax,imax = size(mask)

    sess = tf.Session()

    # # Repeat the input indefinitely.
    # # training dataset iterator
    # train_dataset = tf.data.Dataset.from_generator(
    #     train_datagen, (tf.float32,tf.float32),
    #     (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).repeat().shuffle(shuffle_buffer_size).batch(batch_size)
    # train_iterator = train_dataset.make_one_shot_iterator()
    # train_iterator_handle = sess.run(train_iterator.string_handle())

    # # test dataset without added clouds
    # # must be reinitializable
    # test_dataset = tf.data.Dataset.from_generator(
    #     test_datagen, (tf.float32,tf.float32),
    #     (tf.TensorShape([jmax,imax,nvar]),tf.TensorShape([jmax,imax,2]))).batch(batch_size)

    # test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
    #                                                 test_dataset.output_shapes)
    # test_iterator_init_op = test_iterator.make_initializer(test_dataset)

    # test_iterator_handle = sess.run(test_iterator.string_handle())

    # handle = tf.placeholder(tf.string, shape=[], name = "handle_name_iterator")
    # iterator = tf.data.Iterator.from_string_handle(
    #         handle, train_iterator.output_types, output_shapes = train_iterator.output_shapes)


    # inputs_,xtrue = iterator.get_next()



    # encoder
    enc_nlayers = length(enc_ksize)
    enc_conv = Vector{Any}(undef, enc_nlayers)
    enc_avgpool = Vector{Any}(undef, enc_nlayers)

    enc_avgpool[1] = inputs_

    for l in 1:enc_nlayers
        enc_conv[l] = tf.layers.conv2d(enc_avgpool[l-1],
                                       enc_ksize[l],
                                       (3,3),
                                       padding="same",
                                       activation=tf.nn.leaky_relu)
        print("encoder: output size of convolutional layer: ",l,enc_conv[l].shape)

        enc_avgpool[l] = tf.layers.average_pooling2d(enc_conv[l],
                                                     (2,2),
                                                     (2,2),
                                                     padding="same")

        print("encoder: output size of pooling layer: ",l,enc_avgpool[l].shape)
    end
    enc_last = enc_avgpool[-1]

    # # Dense Layer
    # ndensein = enc_last.shape[1:].num_elements()

    # avgpool_flat = tf.reshape(enc_last, [-1, ndensein])
    # dense_units = [ndensein//5]

    # # default is no drop-out
    # dropout_rate = tf.placeholder_with_default(0.0, shape=())

    # dense = [None] * 5
    # dense[0] = avgpool_flat
    # dense[1] = tf.layers.dense(inputs=dense[0],
    #                            units=dense_units[0],
    #                            activation=tf.nn.relu)
    # dense[2] = tf.layers.dropout(inputs=dense[1], rate=dropout_rate)
    # dense[3] = tf.layers.dense(inputs=dense[2],
    #                            units=ndensein,
    #                            activation=tf.nn.relu)
    # dense[4] = tf.layers.dropout(inputs=dense[3], rate=dropout_rate)


    # dense_2d = tf.reshape(dense[-1], tf.shape(enc_last))

    # ### Decoder
    # dec_conv = [None] * enc_nlayers
    # dec_upsample = [None] * enc_nlayers

    # dec_conv[0] = dense_2d

    # for l in range(1,enc_nlayers):
    #     l2 = enc_nlayers-l
    #     dec_upsample[l] = tf.image.resize_images(
    #         dec_conv[l-1],
    #         enc_conv[l2].shape[1:3],
    #         method=resize_method)
    #     print("decoder: output size of upsample layer: ",l,dec_upsample[l].shape)

    #     # short-cut
    #     if l in skipconnections:
    #         print("skip connection at ",l)
    #         dec_upsample[l] = tf.concat([dec_upsample[l],enc_avgpool[l2-1]],3)
    #         print("decoder: output size of concatenation: ",l,dec_upsample[l].shape)

    #     dec_conv[l] = tf.layers.conv2d(
    #         dec_upsample[l],
    #         enc_ksize[l2-1],
    #         (3,3),
    #         padding="same",
    #         activation=tf.nn.leaky_relu)

    #     print("decoder: output size of convolutional layer: ",l,dec_conv[l].shape)

    # # last layer of decoder
    # xrec = dec_conv[-1]

    # loginvσ2_rec = xrec[:,:,:,1]
    # invσ2_rec = tf.exp(tf.minimum(loginvσ2_rec,10))
    # σ2_rec = sinv(invσ2_rec)
    # m_rec = xrec[:,:,:,0] * σ2_rec


    # σ2_true = sinv(xtrue[:,:,:,1])
    # m_true = xtrue[:,:,:,0] * σ2_true
    # σ2_in = sinv(inputs_[:,:,:,1])
    # m_in = inputs_[:,:,:,0] * σ2_in


    # difference = m_rec - m_true

    # mask_issea = tf.placeholder(
    #     tf.float32,
    #     shape = (mask.shape[0], mask.shape[1]),
    #     name = "mask_issea")

    # # 1 if measurement
    # # 0 if no measurement (cloud or land for SST)
    # mask_noncloud = tf.cast(tf.math.logical_not(tf.equal(xtrue[:,:,:,1], 0)),
    #                         xtrue.dtype)

    # n_noncloud = tf.reduce_sum(mask_noncloud)

    # if truth_uncertain:
    #     # KL divergence between two univariate Gaussians p and q
    #     # p ~ N(σ2_1,\mu_1)
    #     # q ~ N(σ2_2,\mu_2)
    #     #
    #     # 2 KL(p,q) = log(σ2_2/σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
    #     # 2 KL(p,q) = log(σ2_2) - log(σ2_1) + (σ2_1 + (\mu_1 - \mu_2)^2)/(σ2_2) - 1
    #     # 2 KL(p_true,q_rec) = log(σ2_rec/σ2_true) + (σ2_true + (\mu_rec - \mu_true)^2)/(σ2_rec) - 1

    #     cost = (tf.reduce_sum(tf.multiply(
    #         tf.log(σ2_rec/σ2_true) + (σ2_true + difference**2) / σ2_rec,mask_noncloud))) / n_noncloud
    # else:
    #     cost = (tf.reduce_sum(tf.multiply(tf.log(σ2_rec),mask_noncloud)) +
    #         tf.reduce_sum(tf.multiply(difference**2 / σ2_rec,mask_noncloud))) / n_noncloud


    # # L2 regularization of weights
    # if regularization_L2_beta != 0:
    #     trainable_variables   = tf.trainable_variables()
    #     lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in trainable_variables
    #                         if "bias" not in v.name ]) * regularization_L2_beta
    #     cost = cost + lossL2

    # RMS = tf.sqrt(tf.reduce_sum(tf.multiply(difference**2,mask_noncloud))
    #               / n_noncloud)

    # # to debug
    # # cost = RMS

    # if tensorboard:
    #     with tf.name_scope("Validation"):
    #         tf.summary.scalar("RMS", RMS)
    #         tf.summary.scalar("cost", cost)
    #         tf.summary.image("m_rec",tf.expand_dims(
    #             tf.reverse(tf.multiply(m_rec,mask_issea),[1]),-1))
    #         tf.summary.image("m_true",tf.expand_dims(
    #             tf.reverse(tf.multiply(m_true,mask_issea),[1]),-1))
    #         tf.summary.image("sigma2_rec",tf.expand_dims(
    #             tf.reverse(tf.multiply(σ2_rec,mask_issea),[1]),-1))

    # # parameters for Adam optimizer (default values)
    # learning_rate = 1e-3
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-08

    # optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon)
    # gradients, variables = zip(*optimizer.compute_gradients(cost))
    # gradients, _ = tf.clip_by_global_norm(gradients, clip_grad)
    # opt = optimizer.apply_gradients(zip(gradients, variables))

    # dt_start = datetime.now()
    # print(dt_start)

    # if tensorboard:
    #     merged = tf.summary.merge_all()
    #     train_writer = tf.summary.FileWriter(outdir + "/train",
    #                                       sess.graph)
    #     test_writer = tf.summary.FileWriter(outdir + "/test")
    # else:
    #     # unused
    #     merged = tf.constant(0.0, shape=[1], dtype="float32")

    # index = 0

    # sess.run(tf.global_variables_initializer())

    # saver = tf.train.Saver()

    # # loop over epochs
    # for e in range(epochs):

    #     # loop over training datasets
    #     for ii in range(ceil(train_len / batch_size)):

    #         # run a single step of the optimizer
    #         summary, batch_cost, batch_RMS, bs, _ = sess.run(
    #             [merged, cost, RMS, mask_noncloud, opt],feed_dict={
    #                 handle: train_iterator_handle,
    #                 mask_issea: mask,
    #                 dropout_rate: dropout_rate_train})

    #         if tensorboard:
    #             train_writer.add_summary(summary, index)

    #         index += 1

    #         if ii % 20 == 0:
    #             print("Epoch: {}/{}...".format(e+1, epochs),
    #                   "Training loss: {:.4f}".format(batch_cost),
    #                   "RMS: {:.4f}".format(batch_RMS))

    #     if e % save_each == 0:
    #         print("Save output",e)

    #         timestr = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    #         fname = os.path.join(outdir,"data-{}.nc".format(timestr))

    #         # reset test iterator, so that we start from the beginning
    #         sess.run(test_iterator_init_op)

    #         for ii in range(ceil(test_len / batch_size)):
    #             summary, batch_cost,batch_RMS,batch_m_rec,batch_σ2_rec = sess.run(
    #                 [merged, cost,RMS,m_rec,σ2_rec],
    #                 feed_dict = { handle: test_iterator_handle,
    #                               mask_issea: mask })

    #             # time instances already written
    #             offset = ii*batch_size
    #             savesample(fname,batch_m_rec,batch_σ2_rec,meandata,lon,lat,e,ii,
    #                        offset)

    #     if e % save_model_each == 0:
    #         save_path = saver.save(sess, os.path.join(
    #             outdir,"model-{:03d}.ckpt".format(e+1)))

    # dt_end = datetime.now()
    # print(dt_end)
    # print(dt_end - dt_start)
