# Layers in CNTK #
In this Layers folder we'll cover how to implement layers in CNTK.
Let's remember that we have different layer types like:
- A dense (fully connected) layer
- A Convolution layer
- A MaxPooling layer
- A Recurrent layer
- A LSTM layer

>NOTE: Here's some [great documentation](https://www.cntk.ai/pythondocs/layerref.html) regarding layers in CNTK if you want to read more about it.

In this repo we'll cover on how to implement a network using dense layers. But before we start let's remember some code from the previous [DataReaders](https://github.com/JorgeCupi/CNTKSamples/tree/master/python/04_DataReaders) sample:

```python
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.ops import input_variable
path = "dataset.txt"
featuresShapeValue = 6
labelsShapeValue = 1

featuresShape = input_variable(featuresShapeValue)
labelsShape = input_variable(labelsShapeValue)

ctfdResult = CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=featuresShapeValue),
        labels=StreamDef(field='labels', shape=labelsShapeValue)))

reader = MinibatchSource(ctfdResult)
```

We'll also need to import two new modules:
```python
from cntk.layers import Dense
from cntk.ops import relu
```

### What is Dense and relu? ###
#### Dense layers ####
Traditionally known as Fully Connected layers, now they're popularly known as Dense layers. They are layers which basically connect every neuron in one layer to every neuron in the next layer.
#### REctified Linear Units ####
Or **ReLU** is an rectifier activation function, its also known as a ramp function and is defined as:
```console
f(x) = max(0,x)
```
Without a mathematical definition, what it basically does is to return 0 for a negative value and the same value for non-zero values:
![ReLU function graphic](https://upload.wikimedia.org/wikipedia/en/thumb/6/6c/Rectifier_and_softplus_functions.svg/330px-Rectifier_and_softplus_functions.svg.png)

## Implementing our dense layers ##
#### Choosing an optimal hidden layers nodes number ####
First, we'll  define a size for our layer(s):
```python
numberOfHiddenLayerNodes = 4
```
But how do we decide the number of nodes our hidden layer should have? Well, there's not a formal definition or formula to decide this,  altought there's this interesting book called [Neural Network Design](http://hagan.okstate.edu/NNDesign.pdf#page=469) that states:

*In terms of neural networks, the simplest model is the one that contains the smallest number of free parameters(weights and biases), or equivalently, the smallest number of neurons. To find a network that generalizes well, we need to find the simplest network that fits the data.*

Now, there are a couple of good 'rule of thumb' between the AI community:
```console
numberOfHiddenLayerNodes = (inputNodes + outputNodes)/2
or
inputNodes <= numberOfHiddenLayerNodes <= outputNodes 
```

For our specific case, considering we have 6 inputs and 1 output, the result would be around 4. So:
- If we had 2 hidden layers, each layer should have 2 nodes
- Having only one later this one should have 4 nodes

#### Creating our first hidden layer ####
Now we that explained what is Dense, ReLU and the number of nodes in our hidden layers. Creating a hiddenLayer seems even easier:
```python
hiddenLayerOne = Dense(numberOfHiddenLayerNodes, activation=relu)(featuresShape)
```
We just create a new Dense() object called hiddenLayerOne. Notice that the Dense constructor receives two parameters:
- The number of nodes for this layer (in our case this would be 4)
- An activation function (in our case would be a ReLU function)

We'll also pass the **featuresShape** value which means that it's the data that will be feed to the model.

>NOTE. For our small dataset and sample we don't need more than a hidden layer. Remember that less is more and simplicity is key in an AI experiment. So for our experiment we'll leave the number of hidden layers to one.

## 'Chaining' multiple layers ##
So far, we have 3 layers: An input, an output and a hidden layer.
We chained our input layer with our hidden layer when we created the last one (remember we passed a **featuresShape**):

```python
hiddenLayerOne = Dense(numberOfHiddenLayerNodes, activation=relu)(featuresShape)
```

We're basically do the same Dense() object definition with the next differences:
-  We're setting the number of nodes to the **number of labels** we have in our dataset (2)
- We're keeping the **ReLU** activation function
- We're passing the **hiddenLayerOne** to the outputLayer
```python
outputLayer = Dense(labelsShapeValue, activation=relu)(hiddenLayerOne)
```

## Alternative ways to define layers ##
While our tiny sample doesn't need more than one hidden layer there might be times when we'll need to define multiple hidden layers and pass each one of them to a next layer until we reach our output layer. There's a **Sequential()** anotiation to do this in an easier way:
```python
model = Sequential ([
    Dense(numberOfNodesForLayerOne, activation=relu),
    Dense(numberOfNodesForLayerTwo, activation=softmax),
    ...
    Dense(numberOfNodesForLayerN, activation=softmax))
])

## Usage ##
myModel = model(features)
```