package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.core.Activation
import com.cpuheater.deepdive.layers.Dense
import com.cpuheater.deepdive.models.Sequential
import com.cpuheater.deepdive.util.TestSupport
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import com.cpuheater.deepdive.lossfunctions.{CrossEntropy, MSE}



class MNISTSpec extends TestSupport {


  it should "cross entropy mnist" in {

    val input = 28*28
    val hidden = 30
    val output = 10

    val nbOfEpoch = 2

    val learningRate = 0.5
    val batchSize = 10
    val rngSeed = 123

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    val network = Sequential()

    network.add(Dense(nbOutput = hidden, nbInput = input, activation = Activation.Sigmoid))
    network.add(Dense(nbInput = hidden, nbOutput = output, activation = Activation.Sigmoid))

    network.compile(CrossEntropy)

    (0 until nbOfEpoch).map{
      index =>
        println(s"Epoch ${index}")
        network.fit(mnistTrain, learningRate)
    }

    var count = 0
    var total = 0
    while (mnistTest.hasNext) {
      val next = mnistTest.next()
      val feature = next.getFeatures
      val labels = next.getLabels
      (0 until feature.rows()).foreach {
        index =>
          val output = network.predict(feature.getRow(index).T)
          val prediction: Int = Nd4j.argMax(output).getInt(0)
          val hela = labels.getRow(index)
          val label = Nd4j.argMax(labels.getRow(index)).getInt(0)
          if (prediction == label)
            count = count + 1
          total = total +1
      }
    }

    println(s"Training set accuracy: ${count.toFloat/total} %")
  }


  it should "MSE mnist" in {

    val input = 28*28
    val hidden = 30
    val output = 10

    val nbOfEpoch = 2

    val learningRate = 3
    val batchSize = 10
    val rngSeed = 123

    val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

    val network = Sequential()

    network.add(Dense(nbOutput = hidden, nbInput = input, activation = Activation.Sigmoid))
    network.add(Dense(nbInput = hidden, nbOutput = output, activation = Activation.Sigmoid))

    network.compile(MSE)

    (0 until nbOfEpoch).map{
      index =>
        println(s"Epoch ${index}")
        network.fit(mnistTrain, learningRate)
    }

    var count = 0
    var total = 0
    while (mnistTest.hasNext) {
      val next = mnistTest.next()
      val feature = next.getFeatures
      val labels = next.getLabels
      (0 until feature.rows()).foreach {
        index =>
          val output = network.predict(feature.getRow(index).T)
          val prediction: Int = Nd4j.argMax(output).getInt(0)
          val hela = labels.getRow(index)
          val label = Nd4j.argMax(labels.getRow(index)).getInt(0)
          if (prediction == label)
            count = count + 1
          total = total +1
      }
    }

    println(s"Training set accuracy: ${count.toFloat/total} %")
  }


}
