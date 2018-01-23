package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.nn.MaxPool
import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.api.ops.impl.transforms.convolution.Pooling2D
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.ArrayUtil



class MaxPoolSpec extends TestSupport {

  it should "max pool" in {

    val config = MaxPool(height = 4,
      width = 4,
      poolHeight = 2,
      poolWidth = 2,
      stride = 2,
      name = "")

    val batchSize = 2


    val shape = 2 * 3 * 4 * 4
    val x =  Nd4j.linspace(-0.3, 0.4, 2*3*4*4).reshape(2, 3, 4, 4)

    val dout = Nd4j.linspace(-0.3, 0.4, 2*3*2*2).reshape(2, 3, 2, 2)

    val layer = new MaxPoolLayer(config, 1)

    val out = layer.forward(x)

    val GradResult(dpool, _) = layer.backward(x, dout)

    ArrayUtil.equals(dpool.data().asFloat(), Array(0.0,0.0,0.0,0.0,0.0,-0.3,0.0,-0.26956525,0.0,0.0,0.0,0.0,0.0,-0.23913044,0.0,-0.20869568,0.0,0.0,0.0,0.0,0.0,0.065217376,0.0,0.09565216,0.0,0.0,0.0,0.0,0.0,0.12608698,0.0,0.15652177,0.0,0.0,0.0,0.0,0.0,-0.17826086,0.0,-0.1478261,0.0,0.0,0.0,0.0,0.0,-0.11739132,0.0,-0.08695651,0.0,0.0,0.0,0.0,0.0,0.18695654,0.0,0.21739131,0.0,0.0,0.0,0.0,0.0,0.24782607,0.0,0.2782609,0.0,0.0,0.0,0.0,0.0,-0.05652173,0.0,-0.026086956,0.0,0.0,0.0,0.0,0.0,0.004347831,0.0,0.03478262,0.0,0.0,0.0,0.0,0.0,0.30869567,0.0,0.33913046,0.0,0.0,0.0,0.0,0.0,0.36956525,0.0,0.4)) should be(true)
    ArrayUtil.equals(out.data().asFloat(), Array(-0.2631579,-0.24842104,-0.20421052,-0.1894737,-0.14526315,-0.13052632,-0.0863158,-0.071578965,-0.027368426,-0.012631565,0.031578943,0.04631579,0.09052634,0.105263144,0.1494737,0.16421056,0.20842107,0.22315793,0.26736847,0.28210527,0.32631582,0.34105265,0.3852632,0.4)) should be(true)

  }


}
