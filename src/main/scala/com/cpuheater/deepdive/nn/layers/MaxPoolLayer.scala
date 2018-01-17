package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Conv2d, Linear, MaxPool}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.IsMax
import org.nd4j.linalg.api.ops.impl.transforms.convolution.Pooling2D
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class MaxPoolLayer(config: MaxPool,
                   layerNb: Int) extends Layer {


  def params: mutable.Map[String, INDArray] = throw new UnsupportedOperationException()

  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()

  override def name: String = config.name

  override def activationFn: ActivationFn = throw new UnsupportedOperationException()

  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val Array(n, c, h, w) = x.shape()

    val output = Nd4j.createUninitialized(n * c * config.outHeight * config.outWidth)

    Convolution.pooling2D(x,
      config.poolHeight,
      config.poolWidth,
      config.stride, config.stride, 0, 0, true,
      Pooling2D.Pooling2DType.MAX, 0.0, config.outHeight, config.outWidth, output)

    val outputReshaped = output.reshape(n, c, config.outHeight, config.outWidth)

    cache(ParamType.toString(ParamType.X, layerNb)) = x

    output
  }

  override def backward(dout: INDArray, isTraining: Boolean = true): GradResult = {
    val x = cache(ParamType.toString(ParamType.X, layerNb))
    val Array(batchSize, channels, _, _) = x.shape()

    val col6d = Nd4j.create(Array(batchSize,
      channels,
      config.outHeight,
      config.outWidth,
      config.poolHeight,
      config.poolWidth), 'c')

    val col6dPermuted = col6d.permute(0, 1, 4, 5, 2, 3)

    val dout1d = dout.reshape('c', dout.length, 1)

    val col2d = col6d.reshape('c', batchSize*channels * config.outHeight * config.outWidth, config.poolHeight* config.poolWidth)
    Convolution.im2col(x, config.poolHeight, config.poolWidth, config.stride, config.stride, 0, 0, true, col6dPermuted)
    val isMax = Nd4j.getExecutioner.execAndReturn(new IsMax(col2d, 1))
    isMax.muliColumnVector(dout1d)

    val tempEpsilon = Nd4j.create(Array[Int](channels, batchSize, config.height, config.width), 'c')
    val outEpsilon = tempEpsilon.permute(1, 0, 2, 3)
    Convolution.col2im(col6dPermuted, outEpsilon, config.stride, config.stride, 0, 0, config.height, config.width)
    GradResult(outEpsilon)
  }

}
