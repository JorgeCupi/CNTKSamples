# input_variable sample #
import cntk
myFeatures = 7
features = cntk.input_variable(myFeatures)
print(features)

alternativeFeatures = cntk.input_variable(7)
print(alternativeFeatures))

# input_variable with different shapes ##
import cntk
data  = cntk.input_variable(shape=[3,5])

# Using Parameter #
import cntk
data = cntk.parameter(shape=(3,5), init=2)