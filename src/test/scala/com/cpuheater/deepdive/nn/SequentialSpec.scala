package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{Identity, ReLU}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.core.{OldMultiLayerNetwork, OldSolver}
import com.cpuheater.deepdive.nn.layers.CompType
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil

class SequentialSpec extends TestSupport{

  it should "cross entropy 0.9838" in {

    val features =  Nd4j.create(Array(81.46373469,   76.43038776,   74.09187755, -104.66336735, -112.65646939,  -80.41834694)).reshape(2, 3)
    val labels =  Nd4j.create(Array(Array(0,0,0f,0,1,0,0,0,0,0), Array(0,0,0,0f,0,0,0,0,0,1)))


    val dataSet = new DataSet(features, labels)

    val batchSize = 2
    val loss = SoftMaxLoss
    val lr = 1e-3

    val model = Sequential()
      .add(Linear(3, 3, activation = ReLU))
        .add(Linear(3, 3, activation = ReLU))
          .add(Linear(3, 10))
      .build(loss, lr, batchSize, seed=Some(1))

    model.fit(dataSet)

    val params = model.params()
    val ala = params(CompType.print(CompType.B, 3))
    ArrayUtil.equals(params(CompType.print(CompType.W, 1)).data().asFloat(), Array(0.18116832,0.5621731,0.5070054,0.18287283,-0.1497184,-0.5099528,-0.007933383,0.45228204,0.06829335)) should be(true)
    ArrayUtil.equals(params(CompType.print(CompType.B, 1)).data().asFloat(), Array(-7.668929174542427E-5, -1.2034211977152154E-4, -7.620797259733081E-5)) should be(true)
    ArrayUtil.equals(params(CompType.print(CompType.W, 2)).data().asFloat(), Array(0.37703955,-0.18024689,0.38546833,-0.34902474,-0.4488059,0.38130635,-0.037608624,0.487246,-0.37867162)) should be(true)
    ArrayUtil.equals(params(CompType.print(CompType.B, 2)).data().asFloat(), Array(-1.9377178E-4,0.0,-2.418972E-4)) should be(true)
    ArrayUtil.equals(params(CompType.print(CompType.W, 3)).data().asFloat(), Array(0.25613087,0.057081223,-0.011862806,0.121299416,0.009052873,0.32635394,-0.17559326,-0.19772372,-0.34760797,-0.16386786,-0.36746204,0.087155744,-0.08313438,0.50812435,-0.44754678,-0.42846808,0.28476197,0.11724925,-0.27746817,-0.36048618,0.082767874,-0.15791845,-0.1281909,-0.4512719,0.23391193,0.13846499,-0.068738446,-0.03164369,0.25888157,-0.14146225)) should be(true)
    ArrayUtil.equals(params(CompType.print(CompType.B, 3)).data().asFloat(), Array(-4.5980146E-4,-4.89957E-4,-2.6751454E-10,-2.5520706E-6,4.9999997E-4,-4.8167353E-6,-2.3264317E-6,-3.0122696E-11,-4.052943E-5,4.999835E-4)) should be(true)


  }

}
