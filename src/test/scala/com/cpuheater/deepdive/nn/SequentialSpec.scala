package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{Identity, ReLU}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.core.{OldMultiLayerNetwork, OldSolver}
import com.cpuheater.deepdive.nn.layers.ParamType
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
    val ala = params(ParamType.print(ParamType.B, 3))
    ArrayUtil.equals(params(ParamType.print(ParamType.W, 1)).data().asFloat(), Array(-0.054015346,-0.005436676,0.032674875,-0.0718516,0.11479664,0.011154394,0.118019484,-0.08637421,0.059055984)) should be(true)
    ArrayUtil.equals(params(ParamType.print(ParamType.B, 1)).data().asFloat(), Array(-4.8100333E-6,8.964504E-6,2.1727017E-6)) should be(true)
    ArrayUtil.equals(params(ParamType.print(ParamType.W, 2)).data().asFloat(), Array(0.01889134,0.102771536,0.029062808,0.05503489,0.1900736,-0.051578466,-0.06861945,-0.08821113,-0.007827893)) should be(true)
    ArrayUtil.equals(params(ParamType.print(ParamType.B, 2)).data().asFloat(), Array(8.918694E-5,-8.645359E-5,0.0)) should be(true)
    ArrayUtil.equals(params(ParamType.print(ParamType.W, 3)).data().asFloat(), Array(-0.060899522,-0.10935635,0.030858487,-0.04728897,-0.0054429313,-0.11543602,-0.09133812,0.06194661,-0.15984105,0.11299213,0.1356058,-0.08211454,0.09325509,-4.920495E-4,-0.16787094,0.09738797,-0.035208426,-0.124616444,-0.10728555,-0.14021744,0.08181707,0.12429051,0.11740947,0.06699165,-0.082527466,-0.015832765,-0.08669945,0.014345484,-0.09635385,0.09423958)) should be(true)
    ArrayUtil.equals(params(ParamType.print(ParamType.B, 3)).data().asFloat(), Array(-1.9048853E-4,-1.9651007E-4,-1.9713305E-4,-2.1485498E-4,7.9395995E-4,-2.0446419E-4,-1.8620977E-4,-2.146526E-4,-1.9377633E-4,8.041295E-4)) should be(true)


  }

}
