from load_data import *
from SdA import *
import getopt
import cPickle

def test_SdA(finetune_lr=0.1, pretraining_epochs=3,
             pretrain_lr=0.001, training_epochs=5, 
             patch_filename = 'Training_patches.npy', groundtruth_filename = 'training_reshape.npy',
             test_filename = 'Test_patches.npy', testtruth_filename = 'test_reshape.npy', 
             batch_size=100):
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
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    #########################################################
     resumeTraining = False
     opts, arg = getopt.getopt(sys.argv[1:],"rp:")
     for opt, arg in opts:
         if opt == '-r':
             resumeTraining = True                               # make this true to resume training from saved model    
         elif opt == '-p':
             prefix = arg
            
    # if(resumeTraining):
    #     savedModel = file(prefix+'best_valid_momentum.pkl','rb')
    #     genVariables = cPickle.load(savedModel)
    #     print genVariables
    #     (epoch,best_validation_loss,learning_rate,patience,itr) = genVariables
    #     W = cPickle.load(savedModel)
    #     b= cPickle.load(savedModel)
    #     b_prime= cPickle.load(savedModel)
    # else:
    #     epoch,best_validation_loss,learning_rate,patience,itr = [0,numpy.inf,learning_rate,100,0]
    #     W = None
    #     b = None
    #     b_prime= None       
    ##############################

    datasets = load_data('Training_patches.npy','training_reshape.npy','Test_patches.npy','test_reshape.npy')


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
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=22 * 22,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=5
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)
    print 'Length of pretraining function: ', len(pretraining_fns)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        #print i
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                #sprint batch_index
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

            flag = open('flag.pkl','wb')
            cPickle.dump(1,flag, protocol = cPickle.HIGHEST_PROTOCOL)
            flag.close()

            save_valid = open('pre_training.pkl', 'wb')
            #print 'YO! i=',i,' epoch=',epoch,' cost=',numpy.mean(c) 
            #print pretrain_lr
            genVariables = [i, epoch, numpy.mean(c), pretrain_lr]
            cPickle.dump(genVariables,save_valid,protocol = cPickle.HIGHEST_PROTOCOL)
            for j in xrange(len(sda.params)):
                cPickle.dump(sda.params[j].get_value(borrow=True), save_valid, protocol = cPickle.HIGHEST_PROTOCOL)
            save_valid.close()




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
    flag = open('flag.pkl','wb')
    cPickle.dump(2,flag, protocol = cPickle.HIGHEST_PROTOCOL)
    flag.close()
    log_valid_cost=[]

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
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
                    save_file = open('fine_tuning.pkl','wb')
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


if __name__ == '__main__':
    test_SdA()