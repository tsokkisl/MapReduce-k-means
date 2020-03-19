package kmeans

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import scala.collection.mutable.ListBuffer

object kmeans {
  
  /* Return Haversine distance between two points with coordinates (lon, lat) */
  def haversine(lon1: Double, lat1: Double, lon2: Double, lat2: Double) : Double = {
    val dLat = math.toRadians(lat2 - lat1)
    val dLon = math.toRadians(lon2 - lon1)
    val a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(math.toRadians(lat1)) * math.cos(math.toRadians(lat2))
    val c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    6371 * c
  }
  
  /* Return cluster label based on Haversine distance */
  def clusterize(c : ListBuffer[(Int, (Double, Double))], lon2 : Double, lat2 : Double) : Int = {
    var min = 3.4028235E38
    var j = 0
    for (i <- 0 to 4) {
      var h = haversine(c(i)._2._1, c(i)._2._2, lon2, lat2)
      if (h < min) {
        min = h
        j = i
      }
    }
    return c(j)._1
  }
  
  def main(args: Array[String]) {
    
    val lines = Source.fromFile("../data.csv").getLines().take(5)    
    var data : ListBuffer[ListBuffer[String]] = ListBuffer ()
    var centroids : ListBuffer[(Int, (Double, Double))] = ListBuffer()
    var result : ListBuffer[(Int, (BigDecimal, BigDecimal))] = ListBuffer()
    
    print("Enter k : ")
    val k = scala.io.StdIn.readLine().toInt
    
    print("Enter number of iterations : ")
    val it = scala.io.StdIn.readLine().toInt
    
    for (line <- lines) {
      data += line.split(",").to[ListBuffer];
    }
    
    /* Initialize centroids with the first k lines of data */
    for (i <- 0 to k - 1) {
      centroids += ((i + 1, (data(i)(0).toDouble, data(i)(1).toDouble)))
    }
    
    val conf = new SparkConf()
    conf.setAppName("k-means")
    val sc = new SparkContext(conf)
    
    /* Remove invalid values from raw data */
    val rawData = sc.textFile("hdfs://master:9000/data.csv").filter(x => x.split(",")(0) != "0" && x.split(",")(1) != "0")
    
    /* Calculate new centroids */
    for (i <- 0 to it - 1) {
      val rdd1 = rawData.map(x => (clusterize(centroids, x.split(",")(0).toDouble, x.split(",")(1).toDouble), ((x.split(",")(0).toDouble, x.split(",")(1).toDouble), 1)))
      val rdd2 = rdd1.reduceByKey((x, y) => ((x._1._1 + y._1._1, x._1._2 + y._1._2), x._2 + y._2))
      val rdd3 = rdd2.mapValues(x => ((x._1._1 / x._2, x._1._2 / x._2))).sortByKey()
      centroids = rdd3.collect().to[ListBuffer]
    }
    
    /* Output the result */
    for (i <- 0 to k - 1) {
      result += ((centroids(i)._1, (BigDecimal(centroids(i)._2._1), BigDecimal(centroids(i)._2._2))))
      println (result(i))
    }
  }
}