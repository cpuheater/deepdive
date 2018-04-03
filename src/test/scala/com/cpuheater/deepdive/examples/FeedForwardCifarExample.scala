package com.cpuheater.deepdive.examples

import com.cpuheater.deepdive.activations.ReLU
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn._
import com.cpuheater.deepdive.nn.layers.ParamType
import com.cpuheater.deepdive.util.TestSupport
import org.deeplearning4j.datasets.iterator.impl.{CifarDataSetIterator, MnistDataSetIterator}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil

class FeedForwardCifarExample extends TestSupport{

  it should "cross entropy 0.9838" in {


    val height = 32
    val width = 32
    val channels = 3
    val numFilters = 32
    val filterSize = 7
    val stride = 2
    val padding = 1
    val numClasses = 10

    val batchSize = 128
    val rngSeed = 123
    val epochs = 5


    val cifarTrain: DataSetIterator = new CifarDataSetIterator(batchSize, 50, Array[Int](height, width, channels), false, true)
    val e = cifarTrain.totalExamples
    //val l = cifar.next.getLabels
    val cifarTest: DataSetIterator = new CifarDataSetIterator(batchSize, 100, Array[Int](height, width, channels), false, false)


    val loss = SoftMaxLoss
    val lr = 3e-3

    val model = Sequential()
      .add(Conv2d(height = height,
        width = width,
        channels = channels,
        numFilters = numFilters,
        filterHeight = filterSize,
        filterWidth = filterSize,
        stride = 1,
        padding = 3,
        activation = ReLU,
        name = ""))
      .add(MaxPool(
        poolHeight = 2,
        poolWidth = 2,
        stride = 2,
        name = ""))
      .add(Linear(8192, 100))
      .add(Linear(100, numClasses))
      .build(loss, Optimizer.RMSProp(1e-3), seed=Some(1))

    model.fit(cifarTrain, batchSize, epochs)
    println(model.evaluate(cifarTest))


  }

}
