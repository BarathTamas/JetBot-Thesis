import math
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import Callback
import IPython


'''
Code adapted for RNN/LSTM layers and proper checkpointing from:

    https://github.com/mfbalin/Concrete-Autoencoders
'''
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[-1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)
        
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])
        
        self.selections = K.in_train_phase(samples, discrete_logits, training)

        Y = K.dot(X, K.transpose(self.selections))

        
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class CustomCheckpoint(Callback):
    """
    Saves best model with converged selector layer.
    ----------------------
    
    Arguments:
    
    mean_max_target: only save when mean max of probabilities >= mean_max_target
    """

    def __init__(self, mean_max_target=0.99):
        super(CustomCheckpoint, self).__init__()
        self.mean_max_target = mean_max_target
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        
    
    def on_train_begin(self, logs=None):
        self.best_train= np.inf
        self.best_val = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get("val_loss")
        current_train = logs.get("loss")
        mean_max_current = self.get_monitor_value(logs)
        if mean_max_current >= self.mean_max_target and np.less(current_val, self.best_val):
            self.best_val = current_val
            self.best_train= current_train
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            print("New best converged model | loss: %.4f, val_loss: %.4f" % (self.best_train,self.best_val))
            

    def on_train_end(self, logs=None):
        print("Restoring best converged model | loss: %.4f, val_loss: %.4f" % (self.best_train,self.best_val))
        self.model.set_weights(self.best_weights)
            
    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        return monitor_value

class StopperCallback(EarlyStopping):
    
    def __init__(self, mean_max_target = 0.998, out=None):
        self.mean_max_target = mean_max_target
        self.out=out
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
    
    def on_epoch_end(self, epoch, logs = None):
        sep="=================================================================================="
        status='Epoch: %d | mean max of probabilities: %.4f | temperature:  %.4f | loss: %.4f, val_loss: %.4f' % (epoch,
                                                                                                  self.get_monitor_value(logs),
                                                                                                  K.get_value(self.model.get_layer('concrete_select').temp),
                                                                                                  logs['loss'],
                                                                                                  logs['val_loss'])
        self.out.update(IPython.display.Pretty(sep+"\n"+status+'\n'+sep))
    
    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector():
    
    def __init__(self, k, output_function, num_epochs = 300, batch_size = None, learning_rate = 0.001, start_temp = 10.0, min_temp = 0.1, tryout_limit = 5):
        self.k = k
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.out=None
        
    def fit(self, X, Y = None, val_X = None, val_Y = None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)
        
        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)
        
        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size
        
        for i in range(self.tryout_limit):
            
            K.set_learning_phase(1)
            
            inputs = Input(shape = X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))
            
            self.concrete_select = ConcreteSelect(self.k, self.start_temp, self.min_temp, alpha, name = 'concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            self.model = Model(inputs, outputs)

            self.model.compile(Adam(self.learning_rate), loss =root_mean_squared_error)
            
            print(self.model.summary())
            
            self.out = display(IPython.display.Pretty('Start training'), display_id=True)
            
            stopper_callback = StopperCallback(out=self.out)
            checkpoint_callback=CustomCheckpoint(0.99)
            
            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose = 0, callbacks = [stopper_callback, checkpoint_callback],
                                  validation_data = validation_data)#, validation_freq = 10)
            
            if K.get_value(K.mean(K.max(K.softmax(self.concrete_select.logits, axis = -1)))) >= stopper_callback.mean_max_target:
                break
            
            num_epochs *= 2
        
        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
            
        return self
    
    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
    
    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))
    
    def transform(self, X):
        return X[self.get_indices()]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices = False):
        return self.get_indices() if indices else self.get_mask()
    
    def get_params(self):
        return self.model