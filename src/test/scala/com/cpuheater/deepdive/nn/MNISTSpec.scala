package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.Sigmoid
import com.cpuheater.deepdive.lossfunctions.CrossEntropyLoss
import com.cpuheater.deepdive.nn.layers.Dense
import com.cpuheater.deepdive.util.TestSupport
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._



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

    val network = com.cpuheater.deepdive.nn.models.Sequential()

    network.add(Dense(nbOutput = hidden, nbInput = input, activation = Sigmoid))
    network.add(Dense(nbInput = hidden, nbOutput = output, activation = Sigmoid))

    network.compile(CrossEntropyLoss)

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

/*
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

    network.compile(MSELoss)

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
  }*/


}
