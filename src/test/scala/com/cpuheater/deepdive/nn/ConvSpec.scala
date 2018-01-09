package com.cpuheater.deepdive.nn

import com.cpuheater.deepdive.nn.layers.Convolutional
import com.cpuheater.deepdive.util.TestSupport
import org.junit.Assert.assertEquals
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex.point
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex, NDArrayIndexAll}
import org.nd4s.Implicits._



class ConvSpec extends TestSupport {

  it should "linspace" in {

    val x =  Nd4j.linspace(-0.1, 0.5, 2*3*4*4).reshape(2, 3, 4, 4)
    val w =  Nd4j.linspace(-0.2, 0.3, 3*3*4*4).reshape(3, 3, 4, 4)
    val b = Nd4j.linspace(-0.1, 0.2, 3)
    val dout = Nd4j.linspace(-0.1, 0.2, 24).reshape(2, 3, 2, 2)

    val (out, cache) = preOutput(x, w, b, stride = 2, pad = 1)

    val (dx, dw, db) = backprop(dout, cache._1, cache._2, cache._3, cache._4, cache._5, cache._6)
    println(dw)

  }


  def backprop(dout: INDArray, x: INDArray, weights: INDArray, b: INDArray,
               pad: Int, stride: Int, xCols: INDArray): (INDArray, INDArray, INDArray) = {
    val Array(n, c, h, w) = x.shape()
    val Array(nk, _, kh, kw) = weights.shape()

    val oh = (h + 2 * pad - kh) / stride + 1
    val ow = (w + 2 * pad - kw) / stride +1

    val db = dout.sum(0, 2, 3)
    //dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    //dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
    val doutReshaped = dout.permute(1,2,3,0)
      .reshape(nk, dout.size(0) * dout.size(2) * dout.size(3))

    val dw = doutReshaped.dot(xCols.T).reshape(weights.shape(): _*)

    //miniBatch, depth, kH, kW, outH, outW
    val dx2d = weights
      .reshape(weights.size(0),  weights.size(1) * weights.size(2) * weights.size(3)).T.dot(doutReshaped)

    //C * field_height * field_width, -1, N
    val dx6d = dx2d.reshape(c, kh, kw, ow, oh, n)
    //val dx6d = dx2d.reshape(kw, kh, c, ow, oh, n)
    //val eps6dPermuted = dx6d.permute(5, 2, 1, 0, 4, 3)
    val eps6dPermuted = dx6d.permute(5, 0, 1, 2, 4, 3)
    println(eps6dPermuted.shape().toList)


    val epsNextOrig = Nd4j.create(c, n, h, w)


    val epsNext = epsNextOrig.permute(1, 0, 2, 3)

    /**
      *
      * //number of images
        int n = col.size(0);
        //number of columns
        int c = col.size(1);
        //kernel height
        int kh = col.size(2);
        //kernel width
        int kw = col.size(3);
        //out height
        int outH = col.size(4);
        //out width
        int outW = col.size(5);
      *
      */


    val dx = Convolution.col2im(eps6dPermuted,epsNext, stride, stride, pad, pad, h, w)

    (dx, dw, db)
  }


  def preOutput(x: INDArray, weights: INDArray, b: INDArray , stride: Int, pad: Int) = {


    val Array(n, c, h, w) = x.shape()
    val Array(nk, _, kh, kw) = weights.shape()

    if((w + 2 * pad - kw) % stride != 0 || (h + 2 * pad - kh) % stride != 0)
      throw new Exception("Invalid")

    val oh = (h + 2 * pad - kh) / stride + 1
    val ow = (w + 2 * pad - kw) / stride +1

    val x2col =  Convolution.im2col(x, kh, kw, stride, stride, pad, pad, true)
    //miniBatch, depth, kH, kW, outH, outW
    //List(2, 3, 4, 4, 2, 2)
    println(x2col.shape().toList)
    val weightsReshaped = weights.reshape(weights.size(0), kw * kh * weights.size(1)) //.dot(out) + b.reshape(-1, 1)
    val x2colReshaped = x2col.permute(1,2,3, 4,5, 0).reshape(x2col.size(1) * x2col.size(2)* x2col.size(3), x2col.size(0) * x2col.size(5) * x2col.size(4))
    val bb = b.reshape(b.columns(), 1).broadcast(Array(b.columns(), x2col.size(0) * x2col.size(5) * x2col.size(4)): _*)
    val result = weightsReshaped.dot(x2colReshaped) + bb
    val cache = (x, weights, b, pad, stride, x2colReshaped)
    (result.reshape(nk, oh, ow, n).permute(3, 0, 1, 2), cache)

  }



  it should "ula" in {

    val kh: Int = 2
    val kw: Int = 2
    val sy: Int = 2
    val sx: Int = 2
    val ph: Int = 1
    val pw: Int = 1
    val linspaced: INDArray = Nd4j.linspace(0, 15, 16).reshape(1, 1, 4, 4)
    println(linspaced)
    //Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out2p)
    val newTest: INDArray = Convolution.im2col(linspaced, kh, kw, sy, sx, ph, pw, false)


    System.out.println(newTest.shape().toList)
    System.out.println(newTest)

  }




  it should "dupa" in {
    val miniBatch = 2
    val depth = 2
    val height = 3
    val width = 3

    val outH = 2
    val outW = 2
    val kH = 2
    val kW = 2
    val sX = 1
    val sY = 1
    val pX = 0
    val pY = 0


    val input = Nd4j.create(Array[Int](miniBatch, depth, height, width), 'c')
    input.put(Array[INDArrayIndex](point(0), point(0), new NDArrayIndexAll(true), new NDArrayIndexAll(true)), Nd4j.create(Array[Array[Double]](Array(0, 1, 2), Array(3, 4, 5), Array(6, 7, 8))))
    input.put(Array[INDArrayIndex](point(0), point(1), new NDArrayIndexAll(true), new NDArrayIndexAll(true)), Nd4j.create(Array[Array[Double]](Array(9, 10, 11), Array(12, 13, 14), Array(15, 16, 17))))
    input.put(Array[INDArrayIndex](point(1), point(0), NDArrayIndex.all(), NDArrayIndex.all()), Nd4j.create(Array[Array[Double]](Array(18, 19, 20), Array(21, 22, 23), Array(24, 25, 26))))
    input.put(Array[INDArrayIndex](point(1), point(1), NDArrayIndex.all(), NDArrayIndex.all()), Nd4j.create(Array[Array[Double]](Array(27, 28, 29), Array(30, 31, 32), Array(33, 34, 35))))

    //Expected data:
    val expected = Nd4j.create(Array[Int](miniBatch, depth, kH, kW, outH, outW), 'c')

    //Example 0
    //depth 0
    expected.put(Array[INDArrayIndex](point(0), point(0), NDArrayIndex.all, NDArrayIndex.all, point(0), point(0)), Nd4j.create(Array[Array[Double]](Array(0, 1), Array(3, 4))))
    expected.put(Array[INDArrayIndex](point(0), point(0), NDArrayIndex.all, NDArrayIndex.all, point(0), point(1)), Nd4j.create(Array[Array[Double]](Array(1, 2), Array(4, 5))))
    expected.put(Array[INDArrayIndex](point(0), point(0), NDArrayIndex.all, NDArrayIndex.all, point(1), point(0)), Nd4j.create(Array[Array[Double]](Array(3, 4), Array(6, 7))))
    expected.put(Array[INDArrayIndex](point(0), point(0), NDArrayIndex.all, NDArrayIndex.all, point(1), point(1)), Nd4j.create(Array[Array[Double]](Array(4, 5), Array(7, 8))))
    //depth 1
    expected.put(Array[INDArrayIndex](point(0), point(1), NDArrayIndex.all, NDArrayIndex.all, point(0), point(0)), Nd4j.create(Array[Array[Double]](Array(9, 10), Array(12, 13))))
    expected.put(Array[INDArrayIndex](point(0), point(1), NDArrayIndex.all, NDArrayIndex.all, point(0), point(1)), Nd4j.create(Array[Array[Double]](Array(10, 11), Array(13, 14))))
    expected.put(Array[INDArrayIndex](point(0), point(1), NDArrayIndex.all, NDArrayIndex.all, point(1), point(0)), Nd4j.create(Array[Array[Double]](Array(12, 13), Array(15, 16))))
    expected.put(Array[INDArrayIndex](point(0), point(1), NDArrayIndex.all, NDArrayIndex.all, point(1), point(1)), Nd4j.create(Array[Array[Double]](Array(13, 14), Array(16, 17))))

    //Example 1

    expected.put(Array[INDArrayIndex](point(1), point(0), NDArrayIndex.all, NDArrayIndex.all, point(0), point(0)), Nd4j.create(Array[Array[Double]](Array(18, 19), Array(21, 22))))
    expected.put(Array[INDArrayIndex](point(1), point(0), NDArrayIndex.all, NDArrayIndex.all, point(0), point(1)), Nd4j.create(Array[Array[Double]](Array(19, 20), Array(22, 23))))
    expected.put(Array[INDArrayIndex](point(1), point(0), NDArrayIndex.all, NDArrayIndex.all, point(1), point(0)), Nd4j.create(Array[Array[Double]](Array(21, 22), Array(24, 25))))
    expected.put(Array[INDArrayIndex](point(1), point(0), NDArrayIndex.all, NDArrayIndex.all, point(1), point(1)), Nd4j.create(Array[Array[Double]](Array(22, 23), Array(25, 26))))

    expected.put(Array[INDArrayIndex](point(1), point(1), NDArrayIndex.all, NDArrayIndex.all, point(0), point(0)), Nd4j.create(Array[Array[Double]](Array(27, 28), Array(30, 31))))
    expected.put(Array[INDArrayIndex](point(1), point(1), NDArrayIndex.all, NDArrayIndex.all, point(0), point(1)), Nd4j.create(Array[Array[Double]](Array(28, 29), Array(31, 32))))
    expected.put(Array[INDArrayIndex](point(1), point(1), NDArrayIndex.all, NDArrayIndex.all, point(1), point(0)), Nd4j.create(Array[Array[Double]](Array(30, 31), Array(33, 34))))
    expected.put(Array[INDArrayIndex](point(1), point(1), NDArrayIndex.all, NDArrayIndex.all, point(1), point(1)), Nd4j.create(Array[Array[Double]](Array(31, 32), Array(34, 35))))


    println(input.shape().toList)
    val out = Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false)
    assertEquals(expected, out)

    //Now: test with a provided results array, where the results array has weird strides
    val out2 = Nd4j.create(Array[Int](miniBatch, depth, outH, outW, kH, kW), 'c')
    val out2p = out2.permute(0, 1, 4, 5, 2, 3)
    Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out2p)
    assertEquals(expected, out2p)

    val out3 = Nd4j.create(Array[Int](miniBatch, outH, outW, depth, kH, kW), 'c')
    val out3p = out3.permute(0, 3, 4, 5, 1, 2)
    Convolution.im2col(input, kH, kW, sY, sX, pY, pX, false, out3p)
    assertEquals(expected, out3p)

  }





  it should "cross entropy 0.9838" in {

    val input = 3*32*32
    val hidden1 = 100
    val hidden2 = 100
    val output = 10

    val nbOfEpoch = 30

    //val (features, labels) = readFromFile
    //val nbOfExamples = features.rows()


    val features =  Nd4j.create(x).reshape(2, 3)
    val labels =  Nd4j.create(Array(Array(0,0,0f,0,1,0,0,0,0,0), Array(0,0,0,0f,0,0,0,0,0,1)))


    val dataSet = new DataSet(features, labels)
    val lr = 1e-3
    val batchSize = 2

    val network = new Convolutional(List(3, 3), input = 3*3, numClasses = 10)
    network.loss(features, y)
  }


  def readFromFile: (INDArray, INDArray) = {

    val oneHotMap = Nd4j.eye(10)
    val lines = io.Source.fromInputStream(getClass.getResourceAsStream("/cifar4.csv")).getLines()

    val (features, labelsArray) = lines.map { x =>
      val value = x.split(",")
      val features = value.reverse.tail
      val label = value.takeRight(1)
      (features.map(_.toFloat), label.map(_.toFloat))
    }.toArray.foldLeft((Array.empty[Array[Float]], Array.empty[Array[Float]])){
      case ((features, labels), (f, l)) =>
        (features :+ f, labels :+ l)
    }

    val labels = Nd4j.create(features.length, 10)
    val dupa = labelsArray.flatten
    val labelsNDarray = labelsArray.flatten.zipWithIndex.foreach{ case (value, index) => labels(index, ->) =   oneHotMap.getRow(value.toInt)}
    (Nd4j.create(features), labels)
  }




    val x = Array(Array( 0.41794341f  , 1.39710028 ,-1.78590431, -0.70882773 ,-0.07472532 ,-0.77501677 ,
      -0.1497979   , 1.86172902 ,-1.4255293 , -0.3763567  ,-0.34227539 , 0.29490764 ,
      -0.83732373  , 0.95218767 , 1.32931659,  0.52465245),
      Array(-0.14809998 , 0.88953195 , 0.12444653,  0.99109251 , 0.03514666 , 0.26207083 ,
      0.14320173   , 0.90101716 , 0.23185863, -0.79725793 , 0.12001014 ,-0.65679608 ,
      0.26917456   , 0.333667   , 0.27423503,  0.76215717),
      Array(-0.69550058 , 0.29214712 ,-0.38489942,  0.1228747  ,-1.42904497 , 0.70286283 ,
      -0.85850947  , -1.14042979, -1.5853599,7 -0.01530138, -0.32156083,  0.56834936,
      -0.19961722  , 1.27286625 , 1.27292534,  1.58102968),
      Array(-1.75626715 , 0.9217743  ,-0.6753054 , -1.43443616 , 0.47021125 , 0.03196734 ,
      0.04448574   , 0.47824879 ,-2.51335181, -1.15740245 ,-0.70470413 ,-1.04978879 ,
      -1.90795589  , 0.49258765 , 0.83736166, -1.4288134 ),
      Array(-0.18982427 ,-1.14094943 ,-2.12570755, -0.41354791 , 0.44148975 , 0.16411113 ,
      -0.65505065  ,-0.30212765 ,-0.25704466, -0.12841368 , 0.26338593 , 0.1672181  ,
      -0.30871951  ,-1.26754462 ,-0.22319022, -0.82993433),
      Array(-1.11271826 ,-0.44613095 ,-0.40001719,  0.36343905 , 0.94992777 ,-0.32379447 ,
      0.27031704   ,-0.63381148 ,-2.71484268,  0.65576139 ,-1.17004858 , 0.0598685  ,
      -1.64182729  ,-0.28069634 ,-0.67946972, -1.80480094))


  val y = Nd4j.create(Array(Array( 0.41794341,  1.39710028, -1.78590431 ,-0.70882773),
           Array(-0.07472532, -0.77501677, -0.1497979  , 1.86172902),
           Array(-1.4255293 , -0.3763567 , -0.34227539 , 0.29490764),
           Array(-0.83732373,  0.95218767,  1.32931659 , 0.52465245),
           Array(-0.14809998,  0.88953195,  0.12444653 , 0.99109251),
           Array( 0.03514666,  0.26207083,  0.14320173 , 0.90101716)))


}
