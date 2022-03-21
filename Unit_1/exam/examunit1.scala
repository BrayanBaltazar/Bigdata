import org.apache.spark.sql.SparkSession
var spark = SparkSession.builder().getOrCreate()


var df = spark.read.option("header", "true").option("inferSchema","true")csv("G:/COSAS PC/FILES/ESCUELA/DECIMO/DATOS_MASIVOS/PRACTICAS/Netflix_2011_2016.csv")
df.show()

//#3 MUESTRAS LOS NOMBRES DE LAS COLUMNAS
dataframe.columns
//MUESTRA LAS COLUMNAS EN ORDEN

df.columns.sorted()
//#4 MUESTRA EL ESQUEMA
df.printSchema()

//#5 IMPRIME LAS PRIMERAS 5 COLUMNAS
df.select($"Date", $"Open", $"High", $"Low", $"Close").show(5)
//IMPRIME TODAS LAS COLUMNAS
df.select($"Date", $"Open", $"High", $"Low", $"Close").show()

//#6 DESCRIBE EL ARCHIVO CSV QUE SE ESTA UTILIZANDO
df.describe().show()

//#7 CREANDO HV RATIO
var hvratio = df.withColumn("HV Ratio",df("High")/df("Volume"))
hvratio.show()

//#8 PICO MAS ALTO DE LA COLUMNA OPEN
df.select($"Date", $"Open").show(1)

//#9 SIGNIFICADO DE CLOSE CON TUS PALABRAS

//#10 MAXIMO Y MINIMO DE LA COLUMNA VOLUMEN
df.select(max ($"Volume")).show()
df.select(min ($"Volume")).show()

//#11 Con Sintaxis Scala/Spark $ conteste los siguiente:


//A CUANTOS DIAS FUE LA COLUMNA CLOSE INFERIOR A 600?
var a = df.filter($"Close" < 600).count()


//B QUE PORCENTAJE DEL TIEMPO FUE LA COLUMNA HIGH MAYOR A 500
var b = (df.filter($"High" > 500).count()*100)/1260

//C ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
df.select(corr("High", "Volume").alias("correlacion de Pearson")).show()

//D ¿Cuál es el máximo de la columna “High” por año?
df.groupBy(year(df("Date")).alias("Year")).max("High").sort(asc("Year")).show()


//E ¿Cuál es el promedio de columna “Close” para cada mes del calendario?
df.groupBy(month(df("Date")).alias("Month")).avg("Close").sort(asc("Month")).show()
