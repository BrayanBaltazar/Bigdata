![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4Dze8yYYzBPaBVPf7j9Mx9NkHZDDzKXzavCoUnkZuO0xqHG3__mjVJOearB9bEeY4sg&usqp=CAU)
# Instituto Tecnológico de Tijuana
### Nombre de Facultad:
#### Ingeniería Informática.
### Proyecto / Tarea / Práctica:
#### Exam Unit 3
### Materia:
#### Datos Masivos
### Facilitador:
#### Jose Christian Romero Hernandez
### Alumnos:
- Erik Saul Rivera Reyes
- Brayan Baltazar Moreno

### Fecha:
#### Tijuana Baja California a 02 de 06 2022 

### K-means
K-means es un algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en k grupos basándose en sus características. 
El agrupamiento se realiza minimizando la suma de distancias entre cada objeto y el centroide de su grupo o cluster. 
Se suele usar la distancia cuadrática.

### Exam
1. Importar una simple sesión Spark. 
2. Utilice las lineas de código para minimizar errores 
3. Cree una instancia de la sesión Spark 
4. Importar la librería de Kmeans para el algoritmo de agrupamiento. 
5. Carga el dataset de Wholesale Customers Data 
6. Seleccione las siguientes columnas: Fresh, Milk, Grocery, Frozen, Detergents_Paper,  Delicassen y llamar a este conjunto feature_data 
7. Importar Vector Assembler y Vector 
8. Crea un nuevo objeto Vector Assembler para las columnas de caracteristicas como un conjunto de entrada, recordando que no hay etiquetas 
9. Utilice el objeto assembler para transformar feature_data 
10.Crear un modelo Kmeans con K=3 
11.Evalúe los grupos utilizando Within Set Sum of Squared Errors WSSSE e imprima los  centroides.

 
### Example
  
```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema","true")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/Wholesale customers data.csv")

val feature_data = data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")

val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

val features = assembler.transform(feature_data)

val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(features)

val WSSE = model.computeCost(features)
println(s"\n\nSum of Squared Errors = $WSSE\n\n")

println("Cluster Centers: ")
model.clusterCenters.foreach(println)

```

### Code Running
![image](https://user-images.githubusercontent.com/40293937/171579052-ec6b5add-9563-4a81-a87d-9e8029133ae1.png)



### Conclusion
El algoritmo funciona de manera iterativa para asignar cada punto de datos a uno de los grupos K en función de las características que se proporcionan. 
Los puntos de datos se agrupan en función de la similitud de las características. Los resultados del algoritmo de agrupamiento K Means son:

Los centroides de los clústeres K, que puedes ser usados para etiquetar nuevos datos.
Etiquetas para los datos de formación, cada punto de datos se asigna a un único clúster.

<<<<<<< HEAD
###Exam defense
https://www.youtube.com/watch?v=pbAkFT-Pm-w&ab_channel=ERIKSAULRIVERAREYES


=======
### Exam Defense
https://www.youtube.com/watch?v=pbAkFT-Pm-w&ab_channel=ERIKSAULRIVERAREYES

>>>>>>> 6cb6902cd58deadb6bb1c4825b7f9d65619ad763
### References.
[1]https://aprendeia.com/algoritmo-kmeans-clustering-machine-learning/




