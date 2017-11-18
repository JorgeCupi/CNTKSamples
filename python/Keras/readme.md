# Using Keras on CNTK #
## What is Keras ##
[Keras](https://keras.io/) is a Python Deep Learning library that is capable of working on top of Deep Learning toolkits like [CNTK](https://github.com/Microsoft/cntk), [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano).
It does wonderful things for us if we want to accelerate project development by building wrappers overs the backend's code. A key phrase extracted from their own website is: *Being able to go from idea to result with the least possible delay is key to doing good research*.

So why will we learn how to use it? Precisely because of that. If we want to go from zero to hero in a deep learning project, a tool like Keras will be useful for it.

## Understanding Keras ##
Keras' core structure is a **model** which allow us to organize layers. If you took a look at the [Layers post from our CNTK101 repo](https://github.com/JorgeCupi/CNTKSamples/tree/master/python/CNTK101/05_Layers) then you'll have a flashback as Keras also has [Sequential model](https://keras.io/getting-started/sequential-model-guide/) that consists of a stack of layers.

## Using Keras on CNTK ##
While its true that Tensor Flow is the mainstream toolkit in the Deep Learning world we'll learn to use Keras on CNTK given the better performance on some algorithms. You can take a look at [this awesome benchmarking work](http://minimaxir.com/2017/06/keras-cntk/) done by Max Woolf [(@minimaxir)](https://twitter.com/minimaxir).

>NOTE: We'll follow the steps written in the [official Microsoft documentation regarding how to use CNTK as a Keras Backend](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras).

If this is your fist time using CNTK take a look at [this installation process](https://github.com/JorgeCupi/CNTKSamples/tree/master/python).

Once that we have CNTK installed we just need to install keras as well:
```console
pip install keras
```

Now that we have both CNTK and Keras installed it's time to set CNTK as our Keras backend. And there's two ways to do it:
- By creating a keras.json file
- By environment variable

### Creating a keras.json file ###
You should locate the keras.json file at:
```console
C:\Users\{yourUsername}\.keras if you are running Windows

$HOME/.keras if you are running Linux
```
Open the file and just modify the backend variable:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "cntk",
    "image_data_format": "channels_last"
}
```
- floatx defines the default float precision
- epsilon is a numeric constant used to avoid dividing by zero in some operations
- image_data_format specifies which data format convention Keras will follow
- backend defines it Keras will use Tensor Flow, CNTK or Theano
>TIP. You can learn more about this keras.json file configuration at the [backend documentation](https://keras.io/backend/) in Keras official site.

### Using the environment variable ###
For Windows:
```console
set KERAS_BACKEND=cntk 
```
For Linux:
```console
export KERAS_BACKEND=cntk
```

That's it! We are ready to start using Keras over CNTK