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
data.value

# Using Constant #
import cntk
data = cntk.Constant(6,shape = (3,4))
data.value

# Using Record #
import cntk.variables as var
record = var.Record(x = 23, y = 32, z = 55)
# printing the record values #
record.x
record.y
record.z