package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.util.TestSupport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.api.ops.impl.transforms.convolution.Pooling2D
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j



class MaxPoolingSpec extends TestSupport {

  it should "linspace" in {
    val shape = 2 * 3 * 4 * 4
    val x =  Nd4j.linspace(-0.3, 0.4, 2*3*4*4).reshape(2, 3, 4, 4)

    val dout = Nd4j.linspace(-0.3, 0.4, 2*3*2*2).reshape(2, 3, 2, 2)

    val poolH = 2
    val poolW = 2
    val stride = 2

    val (out, cache) = preOutput(x, poolH, poolW, stride)

    val dpool = backprop(dout, x, poolH, poolW, stride)
    print(dpool)


  }


  def preOutput(x: INDArray, poolH: Int, poolW: Int, stride: Int) : (INDArray, INDArray)  = {

    val Array(n, c, h, w) = x.shape()

    val outH = 1 + (h - poolH) / stride
    val outW = 1 + (w - poolW) / stride

    val output = Nd4j.createUninitialized(n * c * outH * outW)

    Convolution.pooling2D(x, poolH, poolH, stride, stride, 0, 0, true, Pooling2D.Pooling2DType.MAX, 0.0, outH, outW, output)

    val outputReshaped = output.reshape(n, c, outH, outW)
    println(output)

    (output, output)
  }


  def backprop(dout: INDArray, x: INDArray, poolW: Int, poolH: Int, stride: Int): INDArray = {
    val Array(n, c, h, w) = x.shape()

    val outH = 1 + (h - poolH) / stride
    val outW = 1 + (w - poolW) / stride

    val col6d = Nd4j.create(Array(n, c, outH, outW, poolH, poolW), 'c')
    val col6dPermuted = col6d.permute(0, 1, 4, 5, 2, 3)

    val dout1d = dout.reshape('c', dout.length, 1)

    val col2d = col6d.reshape('c', n*c * outH * outH, poolH * poolW)
    Convolution.im2col(x, poolH, poolW, stride, stride, 0, 0, true, col6dPermuted)
    val isMax = Nd4j.getExecutioner.execAndReturn(new IsMax(col2d, 1))
    isMax.muliColumnVector(dout1d)

    val tempEpsilon = Nd4j.create(Array[Int](c, n, h, w), 'c')
    val outEpsilon = tempEpsilon.permute(1, 0, 2, 3)
    Convolution.col2im(col6dPermuted, outEpsilon, stride, stride, 0, 0, h, w)
    outEpsilon

  }




}
