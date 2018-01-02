package com.cpuheater.deepdive.nn.layers

import com.cpuheater.deepdive.activations.{ActivationFn, ReLU}
import com.cpuheater.deepdive.nn.{Conv, Linear}
import com.cpuheater.deepdive.nn.layers.CompType.PreOutput
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
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

  override def activationFn: ActivationFn = config.activation

  override def forward(x: INDArray, isTraining: Boolean =  true): INDArray = {
    val x2col =  Convolution.im2col(x,
      config.filterHeight,
      config.filterWidth,
      config.stride,
      config.stride,
      config.padding,
      config.padding, true)

    val Array(batchSize, _, _, _) = x.shape()
    val weightsReshaped = params(CompType.print(CompType.W, layerNb))
      .reshape(config.nbOfFilters, config.filterWidth * config.filterWidth * config.channels) //.dot(out) + b.reshape(-1, 1)
    val x2colReshaped = x2col.permute(1,2,3, 4,5, 0).reshape(x2col.size(1) * x2col.size(2)* x2col.size(3), x2col.size(0) * x2col.size(5) * x2col.size(4))
    val bb = params(CompType.print(CompType.B, layerNb)).reshape(params(CompType.print(CompType.B, layerNb)).columns(), 1).broadcast(Array(params(CompType.print(CompType.B, layerNb)).columns(), x2col.size(0) * x2col.size(5) * x2col.size(4)): _*)
    val preOutput = weightsReshaped.dot(x2colReshaped) + bb

    cache(CompType.print(CompType.PreOutput, layerNb)) = preOutput
    cache(CompType.print(CompType.X, layerNb)) = x

    preOutput.reshape(config.nbOfFilters, config.outHeight, config.outWidth, batchSize).permute(3, 0, 1, 2)
  }

  override def backward(dout: INDArray, isTraining: Boolean = true): (INDArray, INDArray, INDArray) = {
    val preOutput = cache(CompType.print(CompType.PreOutput, layerNb))
    val x = cache(CompType.print(CompType.X, layerNb))
    val w = params(CompType.print(CompType.W, layerNb))
    val b = params(CompType.print(CompType.B, layerNb))

    val preOutputDupl = activationFn.derivative(preOutput.dup())
    val da = preOutputDupl * dout

    val dx = da.dot(w.T).reshape(x.shape(): _*)
    val dw = x.reshape(x.shape()(0), -1).T.dot(da)
    val db = da.sum(0)
    (dx, dw, db)
  }


}
