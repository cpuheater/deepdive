package com.cpuheater.deepdive.examples

import com.cpuheater.deepdive.activations.ReLU
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.{Linear, Optimizer, Sequential}
import com.cpuheater.deepdive.nn.layers.ParamType
import com.cpuheater.deepdive.util.TestSupport
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil

class FeedForwardMnistExample extends TestSupport{

  it should "Accuracy 0.9659" in {


    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val batchSize = 128
    val rngSeed = 123
    val epochs = 1

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    val loss = SoftMaxLoss
    val lr = 3e-3

    val model = Sequential()
      .add(Linear(numRows*numColumns, 1000, activation = ReLU))
      .add(Linear(1000, 1000, activation = ReLU))
      .add(Linear(1000, outputNum))
      .compile(loss, Optimizer.RMSProp(1e-3), seed=Some(1))

    model.fit(mnistTrain, batchSize, epochs)
    println(model.evaluate(mnistTest))


  }

}
