package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{Identity, ReLU}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.util.TestSupport
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex, NDArrayIndexAll}
import org.nd4s.Implicits._


class FullyConnectedCIFARSpec extends TestSupport{


  it should "cifar50-loss: 1.1513407342135907E-4" in {

    val height: Int = 32
    val width: Int = 32
    val channels: Int = 3
    val numLabels: Int = CifarLoader.NUM_LABELS
    val numSamples: Int = 50
    val batchSize: Int = 50



    //val cifar = new CifarDataSetIterator(batchSize, numSamples, Array[Int](height, width, channels), true, true)
    //val cifarEval = new CifarDataSetIterator(batchSize, 100, Array[Int](height, width, channels), true, false)

   // val ala = cifar.next(1)
    val (features, labels) = readFromFile("/cifar50.csv")
    val nbOfExamples = features.rows()


    val loss = SoftMaxLoss
    val lr = 1e-3

    val inputSize=3*32*32


    val hiddenSize = 100

    val model = Sequential()
      .add(Linear(inputSize, hiddenSize , activation = ReLU))
      .add(Linear(hiddenSize, hiddenSize, activation = ReLU))
      .add(Linear(hiddenSize, numLabels))
      .build(loss, lr, batchSize, seed=Some(1), numOfEpoch = 10)

    val reshapedFeatures = features.reshape(batchSize, 3, 32, 32)

    model.fit(new DataSet(reshapedFeatures, labels))

    //println(model.params())

    /*val dataSet = cifar.next(numSamples)
    val features = dataSet.getFeatureMatrix
    val labels = dataSet.getLabels
*/
    val pred = model.predict(features)
    val indicies = Nd4j.argMax(pred, 1)
    val lindicies = Nd4j.argMax(labels, 1)

    println(indicies)
    println(lindicies)

  }

  it should "cross entropy 0.9838" in {

    val height: Int = 32
    val width: Int = 32
    val channels: Int = 3
    val numLabels: Int = CifarLoader.NUM_LABELS
    val numSamples: Int = 4
    val batchSize: Int = 4



    val cifar = new CifarDataSetIterator(batchSize, numSamples, Array[Int](height, width, channels), true, true)
    //val cifarEval = new CifarDataSetIterator(batchSize, 100, Array[Int](height, width, channels), true, false)

    val ala = cifar.next(1)
    val (features, labels) = readFromFile("/cifar4.csv")
    val nbOfExamples = features.rows()


    val loss = SoftMaxLoss
    val lr = 1e-3

    val inputSize=3*32*32


    val hiddenSize = 100

    val model = Sequential()
      .add(Linear(inputSize, hiddenSize , activation = ReLU))
      .add(Linear(hiddenSize, hiddenSize, activation = ReLU))
      .add(Linear(hiddenSize, numLabels))
      .build(loss, lr, batchSize, seed=Some(1), numOfEpoch = 10)


    model.fit(new DataSet(features, labels))

    //println(model.params())

    /*val dataSet = cifar.next(numSamples)
    val features = dataSet.getFeatureMatrix
    val labels = dataSet.getLabels
*/
    val pred = model.predict(features)
    val indicies = Nd4j.argMax(pred, 1)
    val lindicies = Nd4j.argMax(labels, 1)

    println(indicies)
    println(lindicies)

  }

  def readFromFile(name: String): (INDArray, INDArray) = {

    val oneHotMap = Nd4j.eye(10)
    val lines = io.Source.fromInputStream(getClass.getResourceAsStream(name)).getLines()

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
    val dupa = labelsArray.flatten
    val labelsNDarray = labelsArray.flatten.zipWithIndex.foreach{ case (value, index) => labels(index, ->) =   oneHotMap.getRow(value.toInt)}
    (Nd4j.create(features), labels)
  }


}
