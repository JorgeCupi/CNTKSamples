from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, cntk
import cntk

path = "dataset.txt"
features_dimension = 6
labels_dimension = 1
feature = cntk.input_variable(features_dimension)
label = cntk.input_variable(labels_dimension)

ctfdResult = CTFDeserializer(
        path, 
        StreamDefs(
                features=StreamDef(field='features', shape=features_dimension), 
                labels=StreamDef(field='labels', shape=labels_dimension)))

reader = MinibatchSource(ctfdResult)
data = reader.next_minibatch(1)

featuresData = data[reader.streams['features']]
featuresData.asarray()
labelsData = data[reader.streams['labels']]
labelsData.asarray()