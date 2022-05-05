//Practica 3

C:\Spark\spark-3.2.1-bin-hadoop3.2\bin\spark-shell

//Parte 1
val Lista=List ( "rojo", "blanco", "negro")

//Parte 2
val Lista = List("rojo","blanco","negro")
val Lista2 = "verde,amarillo,azul,naranja,perla" :: Lista

//Parte 3
import scala.collection.mutable.ListBuffer

var Lista = new ListBuffer[String]()
Lista +="rojo"
Lista +="blanco"
Lista +="negro"
Lista +="verde"
Lista +="amarillo"
Lista +="azul"
Lista +="naranja"
Lista +="perla"
println(Lista(3) + " " + Lista(4) + " " + Lista(5))

//Parte 4
  var a = 1;
        do
        {
            print(a + " ");
            a = a + 5;
        }while(a <= 1000);


//Parte 5
val x = List(1,3,3,4,6,7,3,7)
x.distinct

//Parte 6

         val mapMut = scala.collection.mutable.Map("Jose" -> 20,
                                                  "Luis" -> 24, 
                                                  "Susana" -> 27)
        println("Mapa: " + mapMut)
         mapMut("Miguel") = 23 
          println("Agregando a Miguel " + mapMut)