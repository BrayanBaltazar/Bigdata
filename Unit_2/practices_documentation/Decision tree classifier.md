![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática y Ingeniería en Tecnologías de la Información y Comunicación.
### Proyecto / Tarea / Práctica:
#### Decision Tree Classifier
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno
- Alonso Villegas Luis Antonio
- Rafael Sanchez Baez
### Fecha:
#### Tijuana Baja California a 07 de 04 2022 

# Árboles de decisión
Los árboles de decisión y sus conjuntos son métodos populares para las tareas de clasificación y regresión del aprendizaje
automático. Los árboles de decisión se usan ampliamente porque son fáciles de interpretar, manejan características categóricas, 
se extienden a la configuración de clasificación multiclase, no requieren escalado de características y pueden capturar no linealidades e 
interacciones de características. Los algoritmos de conjuntos de árboles, como los bosques aleatorios y el impulso, se encuentran entre los 
mejores para las tareas de clasificación y regresión.

Un árbol de decisión tiene una estructura similar a un diagrama de flujo donde un nodo interno representa una característica o atributo, 
la rama representa una regla de decisión y cada nodo u hoja representa el resultado. El nodo superior de un árbol de decisión se conoce como nodo raíz.

#### La idea básica detrás de cualquier problema de árbol de decisión es la siguiente:
- Selecciona el mejor atributo utilizando una medida de selección de atributos o características.
- Haz de ese atributo un nodo de decisión y divide el conjunto de datos en subconjuntos más pequeños.
- Comienza la construcción del árbol repitiendo este proceso recursivamente para cada atributo hasta que una de las siguientes condiciones coincida:
   - Todas las variables pertenecen al mismo valor de atributo.
   - Ya no quedan más atributos
   - No hay más casos.

Aprendizaje basado en árboles de decisión es un método comúnmente utilizado en la minería de datos.El objetivo es crear un modelo que predice el valor de una variable de destino en función de diversas variables de entrada.

![](http://dataanalyticsedge.com/wp-content/uploads/2018/01/1-1.jpg)

Un árbol de decisión es una representación simple para clasificar ejemplos. El aprendizaje basado en árboles de decisión es una de las técnicas más eficaces para la clasificación supervisada. Para esta sección, se supone que todas las funciones tienen dominios discretos finitos, y existe una sola característica de destino llamado la clasificación. Cada elemento del dominio de la clasificación se llama clase. Un árbol de decisión o un árbol de clasificación es un árbol en el que cada nodo interno (no hoja) está etiquetado con una función de entrada. Los arcos procedentes de un nodo etiquetado con una característica están etiquetados con cada uno de los posibles valores de la característica. Cada hoja del árbol se marca con una clase o una distribución de probabilidad sobre las clases.

Un árbol puede ser "aprendido" mediante el fraccionamiento del conjunto inicial en subconjuntos basados en una prueba de valor de atributo. Este proceso se repite en cada subconjunto derivado de una manera recursiva llamada particionamiento recursivo. La recursividad termina cuando el subconjunto en un nodo tiene todo el mismo valor de la variable objetivo, o cuando la partición ya no agrega valor a las predicciones. Este proceso de inducción top-down de los árboles de decisión es un ejemplo de un algoritmo voraz, y es, con mucho, la estrategia más común para aprender árboles de decisión a partir de datos.
En minería de datos, los árboles de decisión se pueden describir también como la combinación de técnicas matemáticas y computacionales para ayudar a la descripción, la categorización y la generalización de un conjunto dado de datos.
Los datos provienen en registros de la forma:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/9354e3bfc0c65eb88a0bf7b6b625dcdbc9e74248)

La variable dependiente, Y, es la variable objetivo que estamos tratando de entender, clasificar o generalizar. El vector x se compone de las variables de entrada, x1, x2, x3 etc., que se utilizan para esa tarea.  
  
### Ejemplo  
  
```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```


### Video explicativo
<https://www.youtube.com/watch?v=JcI5E2Ng6r4&ab_channel=IntuitiveMachineLearning>

<https://www.youtube.com/watch?v=Ih3U8Rju5ck>

### Referencias.
<https://es.wikipedia.org/wiki/Aprendizaje_basado_en_%C3%A1rboles_de_decisi%C3%B3n>
<https://www.youtube.com/watch?v=JcI5E2Ng6r4&ab_channel=IntuitiveMachineLearning>
<https://spark.apache.org/docs/2.4.7/mllib-decision-tree.html>
<https://spark.apache.org/docs/2.4.7/ml-classification-regression.html?fbclid=IwAR3QHShNZQ-gTK3XzVKacVE7NORmYZqX_74qqDw_Yr2lx1sA-nEJJcPh0Kw#decision-tree-classifier>
