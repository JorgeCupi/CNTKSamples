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
    featuresShape: reader.streams.features,
    labelsShape: reader.streams.labels
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
## Understanding the Training session object ##
Funny enough, this class is declared in yet another module: The [cntk.train.training_session](https://www.cntk.ai/pythondocs/cntk.train.training_session.html) module.
What it allows to do is to grab our previously **trainer** object, our **MinibatchSource**, our **input_map**, an **epoch time** and finally train our model. Let's look at the code:

```python
from cntk.train.training_session import training_session
minibatchSize = 50
numberOfSamples = 2208
numberOfSweepsForTraining = 10

trainingSession = training_session(
        trainer=trainer,
        mb_source=reader,
        mb_size=minibatchSize,
        model_inputs_to_streams=input_map,
        max_samples= numberOfSamples * numberOfSweepsForTraining,
        progress_frequency=numberOfSamples
    )

trainingSession.train()
```
The training_session class receives the next parameters:
- trainer = A trainer object to pass
- mb_source = A MinibatchSource object to pass
- mb_size = The size of the samples we want to extract per Minibatch
- model_inputs_to_streams = An input_map that matches our *Labels* and *Features* variables
- max_samples = The maximum number of samples we want to process in the training
- progress_frequency = The frequency that we want for the progress to be printed

Output:
```console
Learning rate per 1 samples: 0.1
Learning rate per 1 samples: 0.01
Learning rate per 1 samples: 0.001
Finished Epoch[1 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.40% * 2000 0.558s (3584.2 samples/s);
Finished Epoch[2 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.00% * 2000 0.062s (32258.1 samples/s);
Finished Epoch[3 of 10]: [Training] loss = 0.000000 * 2000, metric = 34.30% * 2000 0.057s (35087.7 samples/s);
Finished Epoch[4 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.60% * 2000 0.059s (33898.3 samples/s);
Finished Epoch[5 of 10]: [Training] loss = 0.000000 * 2000, metric = 32.50% * 2000 0.056s (35714.3 samples/s);
Finished Epoch[6 of 10]: [Training] loss = 0.000000 * 2000, metric = 32.70% * 2000 0.054s (37037.0 samples/s);
Finished Epoch[7 of 10]: [Training] loss = 0.000000 * 2000, metric = 34.15% * 2000 0.058s (34482.8 samples/s);
Finished Epoch[8 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.15% * 2000 0.057s (35087.7 samples/s);
Finished Epoch[9 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.00% * 2000 0.059s (33898.3 samples/s);
Finished Epoch[10 of 10]: [Training] loss = 0.000000 * 2000, metric = 33.60% * 2000 0.057s (35087.7 samples/s);
```
And what does it mean?
It means that after 10 epochs, our model has an error of 33.39%. Which is not that bad but not good as well. It's not bad because let's remember we put fake data and we 'only' have a dataset of 2000 samples.

To decrease this error there are lots of things we could do:
- Choose a different algorithm
- Change the number of hidden layers / nodes
- Improve the dataset (use real values!)
- Define if we're better with one-hot encoding or not
- A big ETC...