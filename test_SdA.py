from load_data import *
from SdA import *
import getopt
import cPickle
from utils import tile_raster_images
import numpy as np
try:
    import PIL.Image as Image
except ImportError:
    import Image



##########---------SET PREFIX--------##########
Prefix = 'SdA'

root = '../BRATS/3D/'
def test_SdA(finetune_lr=0.1, pretraining_epochs=1,
             pretrain_lr=0.001, training_epochs=1, 
             patch_filename = 'Training_patches.npy', groundtruth_filename = 'Training_labels.npy',
             valid_filename = 'Valid_patches.npy', validtruth_filename = 'Valid_labels.npy',
#             test_filename = root+'Testing_patches.npy', testtruth_filename = root+'Testing_labels.npy',
             batch_size=100, n_ins = 500, n_outs = 5, hidden_layers_sizes = [1000,1000,1000] ):
                 
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations to run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    print '###########################'
    print 'Pretraining epochs: ', pretraining_epochs
    print 'Finetuning epochs: ', training_epochs
    print '###########################'
    
    W = []
    b = []
    
    #########################################################
    #########################################################
    prefix = 'SdA'
    resumeTraining = True
    
    #@@@@@@@@ Needs to be worked on @@@@@@@@@@@@@@@@@
    # Snippet to resume training if the program crashes halfway through #
    opts, arg = getopt.getopt(sys.argv[1:],"rp:")
    for opt, arg in opts:
        if opt == '-r':
            resumeTraining = True                               # make this true to resume training from saved model    
        elif opt == '-p':
            prefix = arg
            
    flagValue = 1    
    
    if(resumeTraining):
        
        flagFile = file(prefix+'flag.pkl','rb')
        
        try:
            flagValue = cPickle.load(flagFile)
        except:
            pass
        
        savedModel_preTraining = file(prefix+'pre_training.pkl','rb')
        genVariables_preTraining = cPickle.load(savedModel_preTraining)
        layer_number, epochs_done_preTraining, mean_cost , pretrain_lr = genVariables_preTraining
        epoch_flag = 1
        print 'Inside resumeTraining!!!!!!!!!!!!!!!!!!'
        no_of_layers = len(hidden_layers_sizes) + 1
        
        for i in xrange(no_of_layers):
            try:
                W.append(cPickle.load(savedModel_preTraining))
                b.append(cPickle.load(savedModel_preTraining))
            except:
                W.append(None)
                b.append(None)
                    
        if flagValue is 2:
            epochFlag_fineTuning = 1
            iterFlag = 1
            savedModel_fineTuning = file(prefix+'fine_tuning.pkl','rb')
            hidden_layers_sizes = cPickle.load(savedModel_fineTuning)
            genVariables_fineTuning = cPickle.load(savedModel_fineTuning)
            epochs_done_fineTuning,best_validation_loss,finetune_lr,patience,iters_done = genVariables_fineTuning
    
   
    else:
        
        layer_number, epochs_done, mean_cost, pretrain_lr = [0,0,0,pretrain_lr]
        epoch_flag = 0
        epochFlag_fineTuning = 0
        iterFlag = 0
        W = None
        b = None
                
    ##############################################################
    ##############################################################

                    
    datasets = load_data(patch_filename,groundtruth_filename,valid_filename,validtruth_filename)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    
#    print 'W: ', W
#    print 'b: ', b
    
    ################################################################
    ################CONSTRUCTION OF SdA CLASS#######################
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=n_outs, W = W, b=b)
        
    print 'SdA constructed'
    ################################################################
    ################################################################
    if flagValue is 1:
    ################################################################
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    
        flag = open(prefix+'flag.pkl','wb')
        cPickle.dump(1,flag, protocol = cPickle.HIGHEST_PROTOCOL)
        flag.close()
            
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
        print 'Length of pretraining function: ', len(pretraining_fns)

        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        log_pretrain_cost = []
        corruption_levels = [.4, .4, .4]
        for i in xrange(sda.n_layers):
        
            if i < layer_number:
                i = layer_number
                #print i
                # go through pretraining epochs
        
            for epoch in xrange(pretraining_epochs):
                ##########################################            
                if epoch_flag is 1 and epoch < epochs_done_preTraining:
                    epoch = epochs_done_preTraining
                    epoch_flag = 0
                    ##########################################
                    # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    #sprint batch_index
                    c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                log_pretrain_cost.append(numpy.mean(c))

            

                save_valid = open(prefix+'pre_training.pkl', 'wb')
                #print 'YO! i=',i,' epoch=',epoch,' cost=',numpy.mean(c) 
                #print pretrain_lr
                genVariables = [i, epoch, numpy.mean(c), pretrain_lr]
                cPickle.dump(genVariables,save_valid,protocol = cPickle.HIGHEST_PROTOCOL)
                for j in xrange(len(sda.params)):
                    cPickle.dump(sda.params[j].get_value(borrow=True), save_valid, protocol = cPickle.HIGHEST_PROTOCOL)
                save_valid.close()
        
        
        pretrain_log_file = open(prefix + 'log_pretrain_cost.txt', "a")
        for l in log_pretrain_cost:
            pretrain_log_file.write("%f\n"%l)
        pretrain_log_file.close()



        #print sda.params[0]
        end_time = time.clock()

        print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
                          
                          
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(datasets=datasets,batch_size=batch_size,learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    flag = open(prefix+'flag.pkl','wb')
    cPickle.dump(2,flag, protocol = cPickle.HIGHEST_PROTOCOL)
    flag.close()
    
    log_valid_cost=[]

    while (epoch < training_epochs) and (not done_looping):
        
        if epochFlag_fineTuning is 1 and epoch < epochs_done_fineTuning:
            epoch = epochs_done_fineTuning
            epochFlag_fineTuning = 0
            
        epoch = epoch + 1
        
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            
            if iterFlag is 1 and iter < iters_done:
                iter = iters_done
                iterFlag = 0
                
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                log_valid_cost.append(this_validation_loss)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    print 'Saving the best validation network'
                    genVariables = [epoch,best_validation_loss,finetune_lr,patience,iter]
                    save_file = open(prefix+'fine_tuning.pkl','wb')
                    cPickle.dump(hidden_layers_sizes, save_file)
                    cPickle.dump(genVariables, save_file)
                    for j in xrange(len(sda.params)):
                        cPickle.dump(sda.params[j].get_value(borrow=True), save_file, protocol = cPickle.HIGHEST_PROTOCOL)
                    save_file.close()
                    valid_file = open('log_valid_cost.txt', "a")
                    for l in log_valid_cost:
                        valid_file.write("%f\n"%l)
                    log_valid_cost=[]


                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    
#    print sda.params
#    
#    image = Image.fromarray(
#        tile_raster_images(X=(sda.params[0]).get_value(borrow=True).T,
#                           img_shape=(22, 22), tile_shape=(10,10),
#                           tile_spacing=(1, 1)))
#    image.save('filters1.png')
    
#    print 'Testing.....'
#    
#    test_data = np.load('Test_patches.npy')
#    test_data = test_data[1]
#    #print test_input
#    
##    layer1 = T.nnet.sigmoid(T.dot(test_input, sda.params[0].get_value(borrow=True).T) + sda.params[1].get_value(borrow=True).T)
##    layer2 = T.nnet.sigmoid(T.dot(layer1,sda.params[2]) + sda.params[3])
##    layer3 = T.nnet.sigmoid(T.dot(layer2,sda.params[4]) + sda.params[5])
##    layer4 = T.nnet.softmax(T.dot(layer3,sda.params[6]) + sda.params[7])
##    
##    pred = T.argmax(layer4, axis = 1)
#    param1 = sda.params[0].get_value(borrow=True)
#    print '###################'
#    print param1
#    param2 = sda.params[1].get_value(borrow=True)
#    print '###################'
#    print param1
#    print '###################'
#    input_patch = T.vector('input_patch')
#    params1 = T.matrix('params1')
#    params2 = T.vector('params2')
#    layer1 = T.nnet.sigmoid(T.dot(input_patch, params1)+params2)
##    layer2 = T.nnet.sigmoid(T.dot(layer1, params[2])+params[3])
##    layer3 = T.nnet.sigmoid(T.dot(layer2, params[4])+params[5])
##    layer4 = T.nnet.softmax(T.dot(layer3, params[6])+params[7])
#    
#    #pred = T.argmax(layer4,axis=1)
#
#    answer = theano.function(inputs=[input_patch, params1, params2], outputs=layer1, allow_input_downcast = True)
#    print answer(test_data,param1, param2)
#    
#    print 'Testing done! Predicted class: '
    


if __name__ == '__main__':
    test_SdA()