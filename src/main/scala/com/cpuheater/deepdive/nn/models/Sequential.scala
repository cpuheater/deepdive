package com.cpuheater.deepdive.nn.models

import com.cpuheater.deepdive.nn.core.FeedForwardNetwork
import com.cpuheater.deepdive.nn.layers.Layer
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.core.FeedForwardNetwork
import com.cpuheater.deepdive.nn.layers.Layer
import com.cpuheater.deepdive.lossfunctions.LossFunction
import com.cpuheater.deepdive.nn.LayerConfig
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class Sequential {

  private var layers = List[LayerConfig]()

  protected var model: FeedForwardNetwork = _

  def add(layer: LayerConfig): Unit = {
    layers = layers :+ layer
  }

  def compile(lossFn: LossFunction/*optimizer: Optimizer*/): Unit = {
    model = ??? //new FeedForwardNetwork(layers, lossFn)
  }

  def fit(dataSet: DataSet, nbEpoch: Int, batchSize: Int, alpha: Double): Unit = {
    model.fit(dataSet, nbEpoch, batchSize, alpha)
  }

  def fit(iterator: DataSetIterator, alpha: Double): Unit = {
    model.fit(iterator, alpha)
  }

  def predict(x: INDArray): INDArray =  {
    model.predict(x)
  }

}

object Sequential {

  def apply() : Sequential = new Sequential()

}
