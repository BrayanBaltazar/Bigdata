![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Exam Unit 2
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno

### Fecha:
#### Tijuana Baja California a 22 de 05 2022 

### Exam
1. Cargar en un dataframe Iris.csv que se encuentra en
https://github.com/jcromerohdz/iris, elaborar la liempieza de datos necesaria para ser
procesado por el siguiente algoritmo (Importante, esta limpieza debe ser por
medio de un script de Scala en Spark) .
a. Utilice la librería Mllib de Spark el algoritmo de Machine Learning multilayer
perceptron
2. ¿Cuáles son los nombres de las columnas?
3. ¿Cómo es el esquema?
4. Imprime las primeras 5 columnas.
5. Usa el metodo describe () para aprender mas sobre los datos del DataFrame.
6. Haga la transformación pertinente para los datos categoricos los cuales serán
nuestras etiquetas a clasificar.
7. Construya el modelo de clasificación y explique su arquitectura.
8. Imprima los resultados del modelo

  
### Code
  
```scala
///librerias a utilizar
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature.StringIndexer 
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

///iniciando una seison de spar
var spark = SparkSession.builder().getOrCreate()

///leer los datos proporcionados
var data = spark.read.option("header", "true").option("inferSchema","true")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/Iris.csv")

///mostrando las columnas
data.columns

///mostrando el esquema
data.printSchema()

///muestra los datos y solamente los primeros 20
data.show()

//describe los datos que ya tenemos proporcionados
data.describe()

//transforma los datos categoricos a numericos
var labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("indexedSpecies").fit(data)
var indexed = labelIndexer.transform(data).withColumnRenamed("indexedSpecies", "label") 

//dividira en un 70% de entrenamiento y un 30% de prueba
var labelIndexer2 = new StringIndexer().setInputCol("label").setOutputCol("indexedSpecies").fit(indexed)
var assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
var  features = assembler.transform(indexed)

//especifica capas de la red neuronal asi como sus caracteristicas
var splits = features.randomSplit(Array(0.7, 0.3), seed = 1234L)
var train = splits(0)
var test = splits(1)
//arreglo de las capas de la red neuronal,en este caso escogemos ciertos valores del mismo arreglo de la capa ya mencionada
var layers = Array[Int](4, 5, 4, 3)

//creacion del modelo de entrenamiento
var trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

//variable para el modelo de entrenamiento
var model = trainer.fit(train)

//nos dara los valores de la precision
var result = model.transform(test)
var predictionAndLabels = result.select("prediction", "label")
var evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

//imprime los valores y detenemos la sesion de spark
println(s"\n\nTest set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


```

### Code Running

1# 
 

2#
 

#3
 

#4
 

#5-8




### Exam defense
https://www.youtube.com/watch?v=RMsvbS5iWr8&ab_channel=ERIKSAULRIVERAREYES
