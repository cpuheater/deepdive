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
    val epochs = 5


    //val cifarTrain: DataSetIterator = new CifarDataSetIterator(batchSize, 50, Array[Int](height, width, channels), false, true)
    //val e = cifarTrain.totalExamples
    //val cifarTest: DataSetIterator = new CifarDataSetIterator(batchSize, 100, Array[Int](height, width, channels), false, false)

    val (features, labels) = readFromFile
    val reshapedFeatures = features.reshape(Array(features.size(0), 3, 32, 32):_*)


    val loss = SoftMaxLoss
    val lr = 1e-3

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
      .add(Linear(8192, 100, ReLU))
      .add(Linear(100, numClasses))
      .build(loss, Optimizer.Adam(lr=0.001), seed=Some(1))

    val cifarTrain = new DataSet(reshapedFeatures, labels)
    model.fit(cifarTrain, batchSize, epochs)
    println(model.evaluate(cifarTrain, 50))

  }



  def readFromFile: (INDArray, INDArray) = {

    val oneHotMap = Nd4j.eye(10)
    val lines = io.Source.fromInputStream(getClass.getResourceAsStream("/cifar50.csv")).getLines()

    val (features, labelsArray) = lines.map { x =>
      val value = x.split(",")
      val features = value.reverse.tail
      val label = value.takeRight(1)
      (features.map(_.toFloat), label.map(_.toFloat))
    }.toArray.foldLeft((Array.empty[Array[Float]], Array.empty[Array[Float]])){
      case ((features, labels), (f, l)) =>
        (features :+ f, labels :+ l)
    }

    val labels = Nd4j.create(features.length, 10)
    val labelsNDarray = labelsArray.flatten.zipWithIndex.foreach{ case (value, index) => labels.putRow(index, oneHotMap.getRow(value.toInt))}
    (Nd4j.create(features), labels)
  }

}
