from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.ops import input_variable, relu
from cntk.layers import Dense
from cntk.metrics import classification_error
from cntk.losses import cross_entropy_with_softmax
from cntk.logging.progress_print import ProgressPrinter
from cntk.learners import learning_rate_schedule, UnitType, adadelta
from cntk.train.trainer import Trainer
from cntk.train.training_session import training_session

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

input_map = {
    featuresShape: reader.streams.features,
    labelsShape: reader.streams.labels
}

numOfEpochs = 10

printer = [ProgressPrinter(
    tag = 'Training',
    num_epochs = numOfEpochs)]

learningRate = learning_rate_schedule([0.1, 0.01, 0.001], UnitType.sample, 700)

trainer = Trainer(outputLayer,(crossEntropy, classificationError), [adadelta(outputLayer.parameters, learningRate)], printer)

minibatchSize = 50
numberOfSamples = 2000
numberOfSweepsForTraining = 5

training_session(
        trainer=trainer,
        mb_source=reader,
        mb_size=minibatchSize,
        model_inputs_to_streams=input_map,
        max_samples=numberOfSamples * numberOfSweepsForTraining,
        progress_frequency=numberOfSamples
    ).train()