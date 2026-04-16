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

# 2. CARGAR EL DATASET
# El CSV tiene un campo 'prices' con JSON embebido que contiene comillas dobles.
# escape='"' + multiLine=True asegura que Spark no confunda esas comillas
# con delimitadores de campo y no desplace las columnas siguientes.
print("\n[1] Cargando dataset...")
df_raw = spark.read.csv(
    "amazon_reviews.csv",
    header=True,
    inferSchema=True,
    quote='"',
    escape='"',
    multiLine=True
)

# Guardar como JSON para cumplir el enunciado
df_raw.write.mode("overwrite").json("amazon_reviews_json")
print("Dataset guardado en formato JSON ✓")

# Recargar desde JSON
df = spark.read.json("amazon_reviews_json")

# Nombres de columnas con punto literal — backticks obligatorios dentro de col()
TEXT_COL   = "`reviews.text`"    # col con punto literal
RATING_COL = "`reviews.rating`" # col con punto literal
ID_COL = "asins"

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
print("\n[5] Consultas distribuidas...")

print("\nPromedio de rating general:")
df.withColumn("rating_num", col(RATING_COL).cast("double")) \
  .filter(col("rating_num").isNotNull()) \
  .select(avg(col("rating_num")).alias("rating_promedio")).show()

print("\nTop 5 reseñas más largas:")
df.withColumn("largo", length(col(TEXT_COL))) \
  .select(ID_COL, "largo") \
  .orderBy(desc("largo")).show(5)

spark.stop()
print("\n✅ Ejercicio 2 completado con éxito.")
