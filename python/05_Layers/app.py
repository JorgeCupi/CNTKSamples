from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs
from cntk.ops import input_variable
from cntk.layers import Dense
from cntk.ops import relu

path = "dataset.txt"
featuresShapeValue = 6
labelsShapeValue = 1

featuresShape = input_variable(featuresShapeValue)
labelsShape = input_variable(labelsShapeValue)

ctfdResult = CTFDeserializer(path, StreamDefs(
        features=StreamDef(field='features', shape=featuresShapeValue),
        labels=StreamDef(field='labels', shape=labelsShapeValue)))

reader = MinibatchSource(ctfdResult)

hiddenLayerDimension = 8
hiddenLayerOne = Dense(hiddenLayerDimension, activation=relu)(featuresShape)
outputLayer = Dense(labelsShapeValue, activation=relu)(hiddenLayerOne)