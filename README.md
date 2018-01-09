# deepdive - Deep Learning library in Scala
Project is using Nd4j for linear algebra.


### Fully Connected example

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