import datasets
import layers
import losses
import models
import activations

model = models.model()
model.add(layers.dense(784, 128))
model.add(activations.relu())
model.add(layers.dense(128, 128))
model.add(activations.relu())
model.add(layers.dense(128, 10))
model.add(activations.softmax())
model.add(losses.crossEntropy())

model.train(datasets.mnist(), epochs=300)