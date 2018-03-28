package com.cpuheater.deepdive.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.util.ArrayUtil
import org.nd4s.Implicits._

object GradientChecking {

  def check(x: INDArray, f: INDArray => INDArray, df: INDArray, epsilon: Double = 1e-5f) = {
    import org.nd4j.linalg.api.iter.NdIndexIterator
    val iter = new NdIndexIterator(x.shape: _*)
    val grad = Nd4j.zerosLike(x)

    while (iter.hasNext) {

      val next = iter.next
      /**
        *
        * oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        */
      val curVal = x.getDouble(next: _*)
      x.putScalar(next, curVal + epsilon)
      val pos = f(x).dup()
      x.putScalar(next, curVal - epsilon)
      val neg = f(x).dup()
      x.putScalar(next, curVal)
      val ula = (pos - neg) * df
      val g = (Nd4j.sum((pos - neg) * df) / (2 * epsilon)).data().getDouble(0)
      grad.putScalar(next, g)
    }

    grad
  }


  /**
    *
    * def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

    */


  def error(a: INDArray, b: INDArray): Double = {
     val eps = Nd4j.zerosLike(a) + 1e-8
     val dupa = abs(a-b)
     val r = Nd4j.max(abs(a-b) / max(eps, abs(a) + abs(b)))
     r.getDouble(0)
  }


}
