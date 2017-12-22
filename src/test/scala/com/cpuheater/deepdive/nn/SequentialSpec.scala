package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{Identity, ReLU}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.core.{MultiLayerNetwork, Solver}
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

class SequentialSpec extends TestSupport{

  it should "cross entropy 0.9838" in {

    val features =  Nd4j.create(Array(81.46373469,   76.43038776,   74.09187755, -104.66336735, -112.65646939,  -80.41834694)).reshape(2, 3)
    val labels =  Nd4j.create(Array(Array(0,0,0f,0,1,0,0,0,0,0), Array(0,0,0,0f,0,0,0,0,0,1)))


    val dataSet = new DataSet(features, labels)

    val batchSize = 2
    val loss = SoftMaxLoss
    val lr = 1e-3

    val network = Sequential(List(Linear(3, 3, activation = ReLU),
                                  Linear(3, 3, activation = ReLU),
                                     Linear(3, 10, activation = Identity)),
                                     loss, lr, batchSize)

    network.fit(dataSet)

  }

}
