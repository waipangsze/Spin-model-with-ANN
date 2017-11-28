******************* For data type **************************
np.savetxt(f, data, fmt='%s', delimiter',') # %s is string, %10.5f is float...
************************************************************

process:
python3 csv2gz.py filename

in run_main.py, change the n_in, n_out, learning_rate, ...



1) pickle
import _pickle as cPickle

Reading MNIST in Python3
MNIST is one of the most well-organized and easy to use datasets that can be used for benchmarking machine learning algorithms. Due its simplicity, this dataset is mainly used as an introductory dataset for teaching machine learning.

The pickle encoded version of this dataset can be downloaded via the link "mnist.pkl.gz". The content of this file is encoded using the python2 pickle and when trying to read this file using python3 we get the following error :


Reading MNIST in Python3
MNIST is one of the most well-organized and easy to use datasets that can be used for benchmarking machine learning algorithms. Due its simplicity, this dataset is mainly used as an introductory dataset for teaching machine learning.

The pickle encoded version of this dataset can be downloaded via the link "mnist.pkl.gz". The content of this file is encoded using the python2 pickle and when trying to read this file using python3 we get the following error :


In [5]: import gzip, pickle
   ...: with gzip.open('mnist.pkl.gz','rb') as ff :
   ...:     train, val, test = pickle.load( ff )
   ...:
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
 in ()
      1 import gzip, pickle
      2 with gzip.open('mnist.pkl.gz','rb') as ff :
----> 3     train, val, test = pickle.load( ff )
      4

UnicodeDecodeError: 'ascii' codec can't decode byte 0x90 in position 614: ordinal not in range(128)
This problem is caused by the difference in encoding codec used by Python3s pickle. To fix this problem, we can read this data as following :


In [6]: import gzip, pickle
   ...: with gzip.open('mnist.pkl.gz','rb') as ff :
   ...:     u = pickle._Unpickler( ff )
   ...:     u.encoding = 'latin1'
   ...:     train, val, test = u.load()
   ...:
The shape of data is as following :


In [8]: print( train[0].shape, train[1].shape )
(50000, 784) (50000,)

In [9]: print( val[0].shape, val[1].shape )
(10000, 784) (10000,)

In [10]: print( test[0].shape, test[1].shape )
(10000, 784) (10000,)

2) For mlp_theano... save parameters;;;;
I also ran into this problem and found this solution. Even if only 'classifier.params' should be necessary, y_pred and input also need to be initialized. Simple way is to store them via pickle and reload them.

Saving:

with open('best_model.pkl', 'wb') as f:
    cPickle.dump((classifier.params, classifier.logRegressionLayer.y_pred, 
                 classifier.input), f)
Predicting function:

def predict(dataset, n_hidden, n_in, n_out):
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')

    # Declare MLP classifier
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )

    # load the saved model
    classifier.params, classifier.logRegressionLayer.y_pred,
        classifier.input = cPickle.load(open('best_model.pkl'))

    predict_model = theano.function(
        inputs=[classifier.input],*emphasized text*
        outputs=classifier.logRegressionLayer.y_pred)

    print("Expected values: ", test_set_y[:10])
    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values:", predicted_values)
Hope this helps.

