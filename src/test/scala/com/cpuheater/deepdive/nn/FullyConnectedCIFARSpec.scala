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
import org.nd4s.Implicits.->

class FullyConnectedCIFARSpec extends TestSupport{

  it should "cross entropy 0.9838" in {

    val height: Int = 32
    val width: Int = 32
    val channels: Int = 3
    val numLabels: Int = CifarLoader.NUM_LABELS
    val numSamples: Int = 1
    val batchSize: Int = 1



    val cifar = new CifarDataSetIterator(batchSize, numSamples, Array[Int](height, width, channels), true, true)
    val cifarEval = new CifarDataSetIterator(batchSize, 100, Array[Int](height, width, channels), true, false)
    val loss = SoftMaxLoss
    val lr = 1e-3

    val inputSize=3*32*32

    val l = cifar.next(batchSize).getLabels
    val hiddenSize = 100

    val model = Sequential()
      .add(Linear(inputSize, hiddenSize , activation = ReLU))
      .add(Linear(hiddenSize, hiddenSize, activation = ReLU))
      .add(Linear(hiddenSize, numLabels))
      .build(loss, lr, batchSize, seed=Some(1), numOfEpoch = 1)

    model.fit(cifar)

    //println(model.params())

    val dataSet = cifar.next(numSamples)
    val features = dataSet.getFeatureMatrix
    val labels = dataSet.getLabels
    val pred = model.predict(features)
    val indicies = Nd4j.argMax(pred, 1)
    val lindicies = Nd4j.argMax(cifar.next(numSamples).getLabels, 1)

    println(indicies)
    println(lindicies)

  }


}
