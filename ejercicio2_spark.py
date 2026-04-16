# ejercicio2_spark.py corregido
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, split, lower, regexp_replace,
    count, avg, desc, length, expr
)

# 1. CREAR SESIÓN DE SPARK
spark = SparkSession.builder \
    .appName("BigData_Ejercicio2") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print(f"Spark versión: {spark.version}")

# ─────────────────────────────────────────
# 2. CARGAR Y LIMPIAR DATASET
# ─────────────────────────────────────────
print("\n[1] Cargando dataset...")
# escape='"' + multiLine=True: el campo 'prices' contiene JSON con comillas dobles
# escapadas (""...""). Sin estas opciones Spark desplaza todas las columnas
# siguientes y 'rating' recibe texto corrupto en vez de números.
df_raw = spark.read.csv(
    "amazon_reviews.csv",
    header=True,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=True
)
df_raw.write.mode("overwrite").json("amazon_reviews_json")

# Recargamos y RENOMBRAMOS las columnas para quitar los puntos
df = spark.read.json("amazon_reviews_json") \
    .withColumnRenamed("reviews.text", "review_text") \
    .withColumnRenamed("reviews.rating", "rating") \
    .withColumnRenamed("reviews.title", "review_title") \
    .withColumnRenamed("name", "producto_nombre") 

# Ahora las variables de columnas son simples y sin comillas raras
TEXT_COL = "review_text"
RATING_COL = "rating"
ID_COL = "asins"
PRODUCT_NAME_COL = "producto_nombre"

print("Columnas listas para procesar ✓")

# 3. MAPREDUCE — WORD COUNT
print("\n[2] MapReduce - Word Count...")

words_df = df.select(
    explode(
        split(
            lower(regexp_replace(col(TEXT_COL), r"[^a-zA-Z\s]", "")),
            r"\s+"
        )
    ).alias("word")
).filter(col("word") != "")

word_count = words_df.groupBy("word").agg(count("*").alias("total"))
print("\nTop 20 palabras más frecuentes:")
word_count.orderBy(desc("total")).show(20)

# 4. BALANCEO DE CARGA
print("\n[3] Análisis de particionamiento...")
distribucion = df.rdd.mapPartitionsWithIndex(
    lambda idx, it: [(idx, sum(1 for _ in it))]
).toDF(["particion", "registros"])
distribucion.show()

# 5. COMPARAR TIEMPOS
print("\n[4] Comparando tiempos de ejecución...")
resultados_tiempo = []
for n_particiones in [2, 4, 8, 16]:
    df_repartido = df.repartition(n_particiones)
    inicio = time.time()
    wc = df_repartido.select(
        explode(split(lower(regexp_replace(col(TEXT_COL), r"[^a-zA-Z\s]", "")), r"\s+"))
    ).count()
    tiempo = fin = time.time() - inicio
    resultados_tiempo.append((n_particiones, tiempo, wc))
    print(f"  Particiones: {n_particiones:>2} | Tiempo: {tiempo:.2f}s")

# 6. CONSULTAS DISTRIBUIDAS

print("\n[5] Consultas distribuidas con DataFrame API...")

# 1. Estadísticas descriptivas de los ratings
print("\n--- Análisis Estadístico de Ratings ---")
df.withColumn("rating_num", expr(f"try_cast({RATING_COL} AS double)")) \
  .select("rating_num").describe().show()

# 2. Top 10 Marcas con mejores reseñas
print("\n--- Top 10 Marcas con mejor calificación promedio ---")
df.withColumn("rating_num", expr(f"try_cast({RATING_COL} AS double)")) \
    .groupBy("brand") \
    .agg(
        avg(col("rating_num")).alias("promedio"),
        count("*").alias("total_resenas")
    ) \
    .filter(col("total_resenas") > 5) \
    .orderBy(desc("promedio")) \
    .show(10)

# 3. Productos con más reseñas
print("\n--- Productos con mayor volumen de interacción ---")
df.groupBy(PRODUCT_NAME_COL).count().orderBy(desc("count")).show(10)
