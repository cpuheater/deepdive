# deepdive 

deepdive - is a deep learning library implemented in Scala, it uses ND4j for linear algebra and signal processing. 



#### Layers
- **Linear**: Linear
  
- **Convolutional**: Convolutional


#### Optimizers

- **SGD**: the stochastic gradient descent
  - *lr*: the learning rate, defaults to 0.01
- **Momentum**: 
  - *lr*: the learning rate, defaults to 0.1
  - *momentum*: decay parameter, defaults to 0.9


#### Fully Connected example

```
val dataSet = new DataSet(features, labels)

val batchSize = 2
val loss = SoftMaxLoss
val lr = 1e-3

val model = Sequential()
      .add(Linear(3, 3, activation = ReLU))
        .add(Linear(3, 3, activation = ReLU))
          .add(Linear(3, 10))
      .build(loss, Optimizer.SGD(1e-3), batchSize, seed=Some(1))

model.fit(dataSet)

```

  <h2 align="center">Work in progress!</h2> 