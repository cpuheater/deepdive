package com.cpuheater.deepdive.nn.serialize

import java.io._
import java.util.zip.{ZipEntry, ZipOutputStream}

import com.cpuheater.deepdive.nn.core.Model
import org.nd4j.linalg.factory.Nd4j


object ModelSerializer {

  private val configFileName = "config.json"

  def save(model: Model, out: String) = {

    val config = model.config
    //val zip = new ZipOutputStream(new FileOutputStream(out))


    /*model.params().map{
      case  (name, array) =>
        val Array(rows, cols) = array.shape()
        rows + cols
    }*/
    //zip.putNextEntry("config.json")
    //zip.write(config)

/*

    val params = new ZipEntry("params.bin")
    zip.putNextEntry(params)
    val dos = new DataOutputStream(new BufferedOutputStream(zip))

    Nd4j.write(model.params, dos)
    finally {
      dos.flush()
      if (!saveUpdater) dos.close()
    }

    zip.close()
*/

  }

  def load(s: String) = {

  }

}
