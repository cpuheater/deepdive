package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.nn.Optimizer.Adam
import com.cpuheater.deepdive.nn.core.{OldMultiLayerNetwork, OldSolver}
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.collection.mutable


class AdamSpec extends TestSupport {

  it should "cross entropy 0.9838" in {

    /**
      *
      * N, D = 4, 5
w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)
      */

    val (n, d) = (4, 5)

    val w =  Nd4j.linspace(-0.4, 0.6, n*d).reshape(n, d)
    val dw =  Nd4j.linspace(-0.6, 0.4, n*d).reshape(n, d)

    val config = Adam()

    val grads = mutable.Map.empty[String, INDArray]
    grads("w1") = Nd4j.zerosLike(dw)

    val adam = new com.cpuheater.deepdive.optimize.Adam(config, grads)
    val newGrad = adam.optimize(w, dw, "w1")
    println(newGrad)


  }

}
