using DINCAE
using Base.Iterators
using Knet
using Dates
using Printf

const F = Float32
Atype = KnetArray{F}
if Atype == Array{F}
    Knet.gpu(false)
else
    Knet.gpu(true)
end

fname = "avhrr_sub_add_clouds.nc"
varname = "SST"

lon,lat,datatime,data_full,missingmask,mask = DINCAE.load_gridded_nc(fname,varname)


#datagen,ntimes,meandata = DINCAE.data_generator(lon,lat,datatime,data_full,missingmask, train = false)
#xin,xtrue = datagen(i);
outdir = "/tmp/DINCAE.jl/"
outdir = "/media/abarth/9982a2e8-599f-4b37-884f-a59aa1c0de80/Alex/Data/DINCAE.jl/Test1"

mkpath(outdir)

                #resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR# ,
                #resize_method = NEAREST_NEIGHBOR# ,
                epochs = 1000# ,
                batch_size = 50# ,
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


    enc_ksize = [nvar]
    append!(enc_ksize,enc_ksize_internal)

dd = DINCAE.NCData(lon,lat,datatime,data_full,missingmask)

i = 1;
xin,xtrue = dd[i];


DINCAE.RVec(dd)

p = partition(DINCAE.RVec(dd),batch_size);
batch = first(p);

inputs_ = cat((b[1] for b in batch)... , dims = Val(4));
xtrue = cat((b[2] for b in batch)... , dims = Val(4));

function iter(lon,lat,datatime,data_full,missingmask,batch_size; train = true)
    dd = DINCAE.NCData(lon,lat,datatime,data_full,missingmask)
    dd.train = train
    p = partition(DINCAE.RVec(dd),batch_size);

    return (
        (Atype(cat((b[1] for b in batch)... , dims = Val(4))), Atype(cat((b[2] for b in batch)... , dims = Val(4)))) for batch in p
    )
end

train = iter(lon,lat,datatime,data_full,missingmask,batch_size; train = true)
test_iter = iter(lon,lat,datatime,data_full,missingmask,batch_size; train = false)

inputs_,xtrue = first(train)

function upsample(x)
    #ratio = (2,2,1,1)
    #return repeat(x,inner = ratio)
    #w = similar(x,2,2,size(x,3),size(x,3))

    w = Atype(zeros(F,2,2,size(x,3),size(x,3)))
    #fill!(w,0)

    for i = 1:size(x,3)
        w[:,:,i,i] .= 1
    end
    return Knet.deconv4(w,x,stride=2)

#    return Knet.deconv4(w,x,stride=2,padding=1)

#    w = Atype(bilinear(F,2,2,size(x,3),size(x,3)))
#    return Knet.deconv4(w,x,stride=2,padding=1)
end


# Define convolutional layer:
struct Conv
    w
    b
    f
end


mse(x,y) = mean((x-y).^2)

(c::Conv)(x) = c.f.(conv4(c.w, x, padding = 1) .+ c.b)
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f)

# Define a chain of layers and a loss function:
struct Chain
    layers
end

function sinv(x; minx = 1e-3)
    return 1 ./ max.(x,minx)
end

function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
#=
    loginvσ2_rec = x[:,:,2:2,:]
    invσ2_rec = exp.(min.(loginvσ2_rec,10))
    σ2_rec = sinv(invσ2_rec)
    m_rec = x[:,:,1:1,:] .* σ2_rec

    return cat(
       m_rec,
       σ2_rec,
    dims = Val(3))=#
    x
end

(c::Chain)(x,y) = mse(c(x),y)

struct Model
    chain
end

function (model::Model)(x)
    x = model.chain(x)

    loginvσ2_rec = x[:,:,2:2,:]
    invσ2_rec = exp.(min.(loginvσ2_rec,10))
    σ2_rec = sinv(invσ2_rec)
    m_rec = x[:,:,1:1,:] .* σ2_rec

    return cat(
       m_rec,
       σ2_rec,
    dims = Val(3))
end

function (model::Model)(inputs_,xtrue)
    xrec = model(inputs_)
    m_rec = xrec[:,:,1:1,:]
    σ2_rec = xrec[:,:,2:2,:]

    σ2_true = sinv(xtrue[:,:,2:2,:])
    m_true = xtrue[:,:,1:1,:] .* σ2_true
    σ2_in = sinv(inputs_[:,:,2:2,:])
    m_in = inputs_[:,:,1:1,:] .* σ2_in


    # # 1 if measurement
    # # 0 if no measurement (cloud or land for SST)
    mask_noncloud = xtrue[:,:,2:2,:] .!= 0

    difference = (m_rec - m_true) .* mask_noncloud
#    @show extrema(Array(m_rec))
#    @show extrema(Array(m_true))
#    @show extrema(Array(difference))

    # mask_issea = tf.placeholder(
    #     tf.float32,
    #     shape = (mask.shape[0], mask.shape[1]),
    #     name = "mask_issea")

    # where there is cloud, σ2_rec_noncloud is 1
    σ2_rec_noncloud = σ2_rec .* mask_noncloud + (1 .- mask_noncloud)

    n_noncloud = sum(mask_noncloud)
    #@show n_noncloud
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
    cost = (sum(log.(σ2_rec_noncloud)) + sum(difference.^2 ./ σ2_rec)) / n_noncloud
#    cost = sum(difference.^2) / n_noncloud

#    cost = sum(σ2_rec .* mask_noncloud)

    return cost
end


struct CatSkip
    inner
end
(m::CatSkip)(x) = cat(m.inner(x), x, dims=Val(3))
#(m::CatSkip)(x) = x




    jmax,imax = size(mask)

#    sess = tf.Session()

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


# Define dense layer:
struct Dense; w; b; f; end
(d::Dense)(x) = dropout(d.f.(d.w * mat(x) .+ d.b),dropout_rate_train)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)


# encoder


ndensein = (size(inputs_,1) * size(inputs_,2)) ÷ (4^length(enc_ksize_internal)) * enc_ksize_internal[end]

@show ndensein


inner = Chain((Dense(ndensein,ndensein ÷ 5),
               Dense(ndensein ÷ 5,ndensein)))

#=
l = length(enc_ksize)-1
m  = CatSkip(inner)

m2 = CatSkip(Chain((
    Conv(3,3,enc_ksize[l],enc_ksize[l+1])
     pool,
m,
upsample,
Conv(3,3,enc_ksize[3],enc_ksize[2]),
=#

model = Model(Chain((Conv(3,3,enc_ksize[1],enc_ksize[2]),
               pool,
               Conv(3,3,enc_ksize[2],enc_ksize[3]),
               pool,

               Conv(3,3,enc_ksize[3],enc_ksize[4]),
               pool,
               Conv(3,3,enc_ksize[4],enc_ksize[5]),
               pool,
               inner,
               x -> reshape(x,(size(inputs_,1) ÷ (2^length(enc_ksize_internal)),
                               size(inputs_,2) ÷ (2^length(enc_ksize_internal)),
                               enc_ksize_internal[end],
                               :)),
               upsample,
               Conv(3,3,enc_ksize[5],enc_ksize[4]),

               upsample,
               Conv(3,3,enc_ksize[4],enc_ksize[3]),
               upsample,
               Conv(3,3,enc_ksize[3],enc_ksize[2]),
               upsample,
               Conv(3,3,enc_ksize[2],2,identity))))


@show size(model(Atype(inputs_)))
@show sum(model(Atype(inputs_)))

#=
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
=#


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

@show model(Atype(inputs_), Atype(xtrue))

losses = []

    # # parameters for Adam optimizer (default values)
    # learning_rate = 1e-3
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-08

    # optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon)
    # gradients, variables = zip(*optimizer.compute_gradients(cost))
    # gradients, _ = tf.clip_by_global_norm(gradients, clip_grad)
    # opt = optimizer.apply_gradients(zip(gradients, variables))

# loop over epochs
@time for e = 1:epochs

    # loop over training datasets
    for (ii,loss) in enumerate(adam(model, train))
        push!(losses,loss)

        if (ii-1) % 20 == 0
            println("epoch: $(@sprintf("%5d",e )) loss $(@sprintf("%5.4f",loss))")

        #             print("Epoch: {}/{}...".format(e+1, epochs),
    #                   "Training loss: {:.4f}".format(batch_cost),
    #                   "RMS: {:.4f}".format(batch_RMS))

        end
    end


    if (e-1) % save_each == 0
        println("Save output $e")

        timestr = Dates.format(Dates.now(),"yyyymmddTHHMMSS")
        fname_rec = joinpath(outdir,"data-$(timestr).nc")

        @time for (ii,(inputs_,xtrue)) in enumerate(test_iter)
            xrec = Array(model(inputs_))
            offset = (ii-1)*batch_size
            DINCAE.savesample(fname_rec,varname,xrec,dd.meandata,lon,lat,ii-1,offset)
        end
    end

    #     if e % save_model_each == 0:
    #         save_path = saver.save(sess, os.path.join(
    #             outdir,"model-{:03d}.ckpt".format(e+1)))

end
