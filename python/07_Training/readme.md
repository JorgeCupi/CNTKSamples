# Training #
As usual, let's recall our source code from previous repos:
```python
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.ops import input_variable, relu
from cntk.layers import Dense
from cntk.metrics import classification_error
from cntk.losses import cross_entropy_with_softmax

path = "dataset.txt"
featuresShapeValue = 6
labelsShapeValue = 1

featuresShape = input_variable(featuresShapeValue)
labelsShape = input_variable(labelsShapeValue)

ctfdResult = CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=featuresShapeValue),
        labels=StreamDef(field='labels', shape=labelsShapeValue)))

reader = MinibatchSource(ctfdResult)

hiddenLayerDimension = 4
hiddenLayerOne = Dense(hiddenLayerDimension, activation=relu)(featuresShape)
outputLayer = Dense(labelsShapeValue, activation=relu)(hiddenLayerOne)

crossEntropy = cross_entropy_with_softmax(outputLayer, labelsShape)
classificationError = classification_error(outputLayer, labelsShape)
```

We're finally reaching the most entertaining part from this mini repo series: Training time!
But let's go step by step, before performing a training time we need to define an input_map object, which is just a dictionary that contains input and output training pairs:
```python
input_map = {
    feature: reader.streams.features,
    label: reader.streams.labels
}
```
With our input map in place we'll create a ProgressPrinter object located in the [cntk.logging](https://www.cntk.ai/pythondocs/cntk.logging.html) package and the [progress_print](https://www.cntk.ai/pythondocs/cntk.logging.progress_print.html) sub module. It allows us to print metrics like our classification and loss errors while the model is training:
```python
from cntk.logging.progress_print import ProgressPrinter
numOfEpochs = 10

printer = [ProgressPrinter(
    tag = 'Training',
    num_epochs = numOfEpochs)]
```
What do tag and num_epochs do?
- tag = It's just a string that prepends the minibatch log lines
- num_epochs = It defines the total number of epochs to be trained. It is optional and used for some metadata

## Understanding the Learning rate schedule ##
Defined in the [cntk.learners module](https://www.cntk.ai/pythondocs/_modules/cntk/learners.html) among other training functions we'll use, The learning_rate_schedule class allows us to vary the learning rate (which tends to be reduced as the number of epochs increase).It has three main parameters: A list of learning rates, a unit type and a an optional epoch size:
```python
from cntk.learners import learning_rate_schedule, UnitType

learningRate = learning_rate_schedule([0.1, 0.01, 0.001], UnitType.sample, 700)
```
This code above would mean that our learning rate will have the next values:
- 0.1 for the first 700 samples
- 0.01 for the next 700 samples
- 0.001 for the next 700 samples

If we wanted to use just a single learning rate for our dataset we would write this instead:
```python
from cntk.learners import learning_rate_schedule, UnitType

learningRate = learning_rate_schedule(1, UnitType.sample)
```

## Understanding the trainer object ##
Located in the [cntk.train.trainer](https://www.cntk.ai/pythondocs/cntk.train.trainer.html) module, this class do what it says: it performs training on our model based on the next parameters:
- An output_layer operation
- Loss and error functions
- A metric we'll track during training
- A learning rate
- A writer

So our trainer object will look like this:
```python
from cntk.train.trainer import Trainer
from cntk.learners import adadelta
trainer = Trainer(outputLayer,(crossEntropy, classificationError), [adadelta(outputLayer.parameters, learningRate)], printer)
```
