package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Conv, Linear}
import com.cpuheater.deepdive.nn.layers.ParamType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable

class ConvLayer(config: Conv,
                override val params: mutable.Map[String, INDArray], layerNb: Int) extends Layer {


  require((config.width + 2 * config.padding - config.filterWidth) % config.stride == 0, "Invalid weidth")
  require((config.height + 2 * config.padding - config.filterHeight) % config.stride == 0, "invalid height")


  private val cache: mutable.Map[String, INDArray] = mutable.Map[String, INDArray]()


  override def name: String = config.name

  override def activationFn: ActivationFn = throw new UnsupportedOperationException()


  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val x2col =  Convolution.im2col(x,
      config.filterHeight,
      config.filterWidth,
      config.stride,
      config.stride,
      config.padding,
      config.padding, true)

    val Array(batchSize, _, _, _) = x.shape()
    val weightsReshaped = params(ParamType.toString(ParamType.W, layerNb))
      .reshape(config.nbOfFilters, config.filterWidth * config.filterWidth * config.channels) //.dot(out) + b.reshape(-1, 1)
    val x2colReshaped = x2col.permute(1,2,3, 4,5, 0).reshape(x2col.size(1) * x2col.size(2)* x2col.size(3), x2col.size(0) * x2col.size(5) * x2col.size(4))
    val bb = params(ParamType.toString(ParamType.B, layerNb)).reshape(params(ParamType.toString(ParamType.B, layerNb)).columns(), 1).broadcast(Array(params(ParamType.toString(ParamType.B, layerNb)).columns(), x2col.size(0) * x2col.size(5) * x2col.size(4)): _*)
    val preOutput = weightsReshaped.dot(x2colReshaped) + bb

    cache(ParamType.toString(ParamType.PreOutput, layerNb)) = preOutput
    cache(ParamType.toString(ParamType.X, layerNb)) = x
    cache(ParamType.toString(ParamType.X2Cols, layerNb)) = x2colReshaped

    preOutput.reshape(config.nbOfFilters, config.outHeight, config.outWidth, batchSize).permute(3, 0, 1, 2)
  }

  override def backward(dout: INDArray, isTraining: Boolean = true): (INDArray, INDArray, INDArray) = {
    val preOutput = cache(ParamType.toString(ParamType.PreOutput, layerNb))
    val x = cache(ParamType.toString(ParamType.X, layerNb))
    val w = params(ParamType.toString(ParamType.W, layerNb))
    val b = params(ParamType.toString(ParamType.B, layerNb))
    val x2cols = cache(ParamType.toString(ParamType.X2Cols, layerNb))
    val Array(batchSize, _, _, _) = x.shape()

    val db = dout.sum(0, 2, 3)
    val doutReshaped = dout.permute(1,2,3,0)
      .reshape(config.nbOfFilters, dout.size(0) * dout.size(2) * dout.size(3))

    val dw = doutReshaped.dot(x2cols.T).reshape(w.shape(): _*)

    val dx2d = w
      .reshape(w.size(0),  w.size(1) * w.size(2) * w.size(3)).T.dot(doutReshaped)

    val dx6d = dx2d.reshape(config.channels,
      config.filterHeight,
      config.filterWidth,
      config.outHeight, config.outWidth, batchSize)

    val eps6dPermuted = dx6d.permute(5, 0, 1, 2, 4, 3)
    val epsNextOrig = Nd4j.create(config.channels, batchSize, config.height, config.width)

    val epsNext = epsNextOrig.permute(1, 0, 2, 3)
    val dx = Convolution.col2im(eps6dPermuted,
      epsNext,
      config.stride,
      config.stride,
      config.padding,
      config.padding, config.height, config.width)

    (dx, dw, db)

  }

  def backprop(dout: INDArray, x: INDArray, weights: INDArray, b: INDArray,
               pad: Int, stride: Int, xCols: INDArray): (INDArray, INDArray, INDArray) = {
    val Array(n, c, h, w) = x.shape()
    val Array(nk, _, kh, kw) = weights.shape()

    val oh = (h + 2 * pad - kh) / stride + 1
    val ow = (w + 2 * pad - kw) / stride +1

    val db = dout.sum(0, 2, 3)
    val doutReshaped = dout.permute(1,2,3,0)
      .reshape(nk, dout.size(0) * dout.size(2) * dout.size(3))

    val dw = doutReshaped.dot(xCols.T).reshape(weights.shape(): _*)

    val dx2d = weights
      .reshape(weights.size(0),  weights.size(1) * weights.size(2) * weights.size(3)).T.dot(doutReshaped)

    val dx6d = dx2d.reshape(c, kh, kw, ow, oh, n)
    val eps6dPermuted = dx6d.permute(5, 0, 1, 2, 4, 3)
    val epsNextOrig = Nd4j.create(c, n, h, w)
    val epsNext = epsNextOrig.permute(1, 0, 2, 3)
    val dx = Convolution.col2im(eps6dPermuted,epsNext, stride, stride, pad, pad, h, w)

    (dx, dw, db)
  }


}
