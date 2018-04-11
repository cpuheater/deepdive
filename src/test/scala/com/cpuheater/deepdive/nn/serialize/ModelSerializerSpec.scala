package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.activations.{Identity, ReLU}
import com.cpuheater.deepdive.lossfunctions.SoftMaxLoss
import com.cpuheater.deepdive.nn.Optimizer.SGD
import com.cpuheater.deepdive.nn.layers.ParamType
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil

class SequentialSpec extends TestSupport{

  it should "test model" in {


    val batchSize = 2
    val epochs = 2
    val loss = SoftMaxLoss
    val lr = 1e-3

    val model = Sequential()
      .add(Linear(3, 3, activation = ReLU))
      .add(Linear(3, 3, activation = ReLU))
      .add(Linear(3, 10))
      .compile(loss, Optimizer.SGD(1e-3), seed=Some(1))

    Sequential.save(model, "/home/luke/tmp/myfile.zip")
  }

}
