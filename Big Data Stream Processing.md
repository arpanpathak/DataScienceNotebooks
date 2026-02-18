## Apache Spark in Kotlin

Spark's primary API is Scala, but it also offers a fully supported Java API. Kotlin interoperates perfectly with Java, so we can use the Java API directly while writing Kotlin code. We'll use the `spark-sql_2.12` and `spark-mllib_2.12` dependencies.

### Example: Feature Engineering and Random Forest Training

```kotlin
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, when, udf}
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Define a data class to represent a row (optional, for type safety)
data class Transaction(
    val userId: String,
    val amount: Double,
    val category: String,
    val label: Double
)

fun main() {
    // 1. Create SparkSession with adaptive query execution enabled
    val spark = SparkSession.builder()
        .appName("FeatureEngineering")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .master("local[*]")  // for local testing; remove in cluster
        .getOrCreate()

    // 2. Read data from Parquet (or CSV, JSON, etc.)
    //    The DataFrame is of type Dataset<Row>
    val df = spark.read().parquet("s3://data/raw/transactions.parquet")

    // 3. Print schema and basic statistics
    df.printSchema()
    df.describe("amount").show()

    // 4. Drop rows with nulls in essential columns
    val cleanDf = df.na().drop(Array("userId", "amount"))

    // 5. Feature engineering: create a new column using when/otherwise
    val withHighValue = cleanDf.withColumn(
        "isHighValue",
        when(col("amount").gt(1000), 1).otherwise(0)
    )

    // 6. User-defined function (UDF) to map category to numeric score
    val categoryToScore = udf(
        { category: String ->
            when (category) {
                "A" -> 1.0
                "B" -> 2.0
                "C" -> 3.0
                else -> 0.0
            }
        },
        DataTypes.DoubleType
    )

    val withCategoryScore = withHighValue.withColumn(
        "categoryScore",
        categoryToScore.apply(col("category"))
    )

    // 7. Assemble features into a vector
    val assembler = VectorAssembler()
        .setInputCols(arrayOf("amount", "categoryScore", "isHighValue"))
        .setOutputCol("features")

    val vectorDf = assembler.transform(withCategoryScore)

    // 8. Scale features
    val scaler = StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
        .setWithStd(true)
        .setWithMean(true)

    val scalerModel = scaler.fit(vectorDf)
    val scaledDf = scalerModel.transform(vectorDf)

    // 9. Split into train/test
    val Array(train, test) = scaledDf.randomSplit(doubleArrayOf(0.8, 0.2), 42L)

    // 10. Train Random Forest classifier
    val rf = RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("scaledFeatures")
        .setNumTrees(50)
        .setMaxDepth(5)

    val model = rf.fit(train)

    // 11. Make predictions and evaluate
    val predictions = model.transform(test)
    predictions.select("label", "prediction").show(5)

    val evaluator = MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Accuracy = $accuracy")

    // 12. Save model
    model.save("s3://models/rf_model_v1")

    spark.stop()
}
```

**Explanation of key parts:**

- **SparkSession builder:** We enable adaptive query execution (`spark.sql.adaptive.enabled`) to let Spark optimize shuffle partitions and skew joins at runtime.
- **Data cleaning:** `na().drop()` removes rows with nulls in critical columns.
- **Column operations:** `withColumn` and `when/otherwise` are used to create new columns (equivalent to PySpark's `withColumn` and `when`).
- **UDFs:** In Kotlin, we define a UDF by passing a lambda to `udf()` and explicitly specifying the return type. The lambda must be serializable.
- **VectorAssembler and StandardScaler:** These are MLlib transformers that operate on DataFrames. We set input/output columns using setters.
- **RandomForestClassifier:** Configured with parameters, then `fit()` returns a model.
- **Evaluation:** `MulticlassClassificationEvaluator` computes accuracy.

**Idiomatic Kotlin touches:**
- Use of Kotlin's lambda syntax for UDF.
- Type-safe array creation with `arrayOf()`.
- Data class `Transaction` can be used to convert Rows to objects (not shown but possible with `map` and encoders).
- Extension functions could be written to chain transformations, but we kept it straightforward for clarity.

---

## Apache Flink in Kotlin

Flink also provides a rich Java API. We'll use the DataStream API with event time processing and windowing. The example below simulates a stream of user clicks, computes page views per 10-minute tumbling window, and writes results to MySQL. We'll use Flink's `Table` API for the SQL-like window query.

Add dependencies: `flink-java`, `flink-streaming-java`, `flink-table-api-java-bridge`, `flink-connector-kafka`, `flink-connector-jdbc`, etc.

### Example: Streaming Page Views with Event Time

```kotlin
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment
import org.apache.flink.table.api.DataTypes
import org.apache.flink.table.api.Expressions.*
import org.apache.flink.table.api.Tumble
import java.time.Duration

// Data class for input events
data class ClickEvent(
    val userId: String,
    val page: String,
    val clickTime: Long  // epoch milliseconds
)

fun main() {
    // 1. Set up streaming environment
    val env = StreamExecutionEnvironment.getExecutionEnvironment()
    val tableEnv = StreamTableEnvironment.create(env)

    // 2. Define source table (Kafka) using SQL DDL
    val sourceDDL = """
        CREATE TABLE user_clicks (
            userId STRING,
            page STRING,
            clickTime TIMESTAMP(3),
            WATERMARK FOR clickTime AS clickTime - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'clicks',
            'properties.bootstrap.servers' = 'kafka:9092',
            'format' = 'json',
            'scan.startup.mode' = 'latest-offset'
        )
    """.trimIndent()
    tableEnv.executeSql(sourceDDL)

    // 3. Define sink table (MySQL)
    val sinkDDL = """
        CREATE TABLE page_views (
            page STRING,
            window_start TIMESTAMP(3),
            window_end TIMESTAMP(3),
            view_count BIGINT
        ) WITH (
            'connector' = 'jdbc',
            'url' = 'jdbc:mysql://mysql:3306/stats',
            'table-name' = 'page_views',
            'username' = 'user',
            'password' = 'pass'
        )
    """.trimIndent()
    tableEnv.executeSql(sinkDDL)

    // 4. Execute streaming query: 10-minute tumbling windows
    val result = tableEnv.sqlQuery("""
        SELECT 
            page,
            TUMBLE_START(clickTime, INTERVAL '10' MINUTE) AS window_start,
            TUMBLE_END(clickTime, INTERVAL '10' MINUTE) AS window_end,
            COUNT(*) AS view_count
        FROM user_clicks
        GROUP BY page, TUMBLE(clickTime, INTERVAL '10' MINUTE)
    """)

    // 5. Insert results into sink
    result.executeInsert("page_views")

    // 6. Execute the Flink job
    env.execute("Page Views Streaming Job")
}
```

**Explanation:**

- **Watermark strategy:** The `WATERMARK` clause defines that events can be up to 5 seconds late. Flink will track event time using the `clickTime` field and generate watermarks accordingly.
- **Source table:** Reads JSON messages from Kafka topic `clicks`. The schema matches our `ClickEvent` data class.
- **Sink table:** JDBC connector writes to a MySQL table.
- **Window query:** `TUMBLE` groups events into 10-minute fixed windows based on `clickTime`. The query computes counts per page per window.
- **Execution:** `executeInsert` runs the streaming insert; `env.execute` starts the job.

If you prefer the DataStream API directly (without SQL), here's a version using Kotlin lambdas:

```kotlin
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows
import org.apache.flink.streaming.api.functions.timestamps.BoundedOutOfOrdernessTimestampExtractor
import org.apache.flink.api.common.eventtime.WatermarkStrategy
import org.apache.flink.api.common.typeinfo.Types
import org.apache.flink.streaming.api.datastream.DataStream
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer
import org.apache.flink.streaming.connectors.jdbc.JdbcSink
import org.apache.flink.streaming.connectors.jdbc.JdbcConnectionOptions
import java.time.Duration
import java.util.Properties

data class ClickEvent(val userId: String, val page: String, val clickTime: Long)
data class PageViewCount(val page: String, val windowStart: Long, val windowEnd: Long, val count: Long)

fun main() {
    val env = StreamExecutionEnvironment.getExecutionEnvironment()

    // Kafka consumer config
    val props = Properties().apply {
        setProperty("bootstrap.servers", "kafka:9092")
        setProperty("group.id", "flink-consumer")
    }

    val kafkaConsumer = FlinkKafkaConsumer("clicks", 
        org.apache.flink.api.common.serialization.SimpleStringSchema(), props)

    // Convert JSON string to ClickEvent (using a simple map; in practice use a JSON parser)
    val clickStream: DataStream<ClickEvent> = env
        .addSource(kafkaConsumer)
        .map { value -> 
            // Assume JSON like {"userId":"u1","page":"home","clickTime":1634567890000}
            // For simplicity, we use a regex or a proper JSON library. Here we hardcode a parser.
            // In production, use Jackson or Gson.
            val json = org.json.JSONObject(value)
            ClickEvent(json.getString("userId"), json.getString("page"), json.getLong("clickTime"))
        }

    // Assign watermarks for event time (5-second out-of-order tolerance)
    val withTimestamps = clickStream.assignTimestampsAndWatermarks(
        WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(5))
            .withTimestampAssigner { event, _ -> event.clickTime }
    )

    // Key by page and open tumbling window of 10 minutes
    val windowed = withTimestamps
        .keyBy { it.page }
        .window(TumblingEventTimeWindows.of(Time.minutes(10)))
        .apply { (page, window, events, out: Collector<PageViewCount>) ->
            out.collect(PageViewCount(page, window.start, window.end, events.count()))
        }

    // Sink to MySQL using JDBC
    val jdbcSink = JdbcSink.sink(
        "INSERT INTO page_views (page, window_start, window_end, view_count) VALUES (?, ?, ?, ?)",
        { ps, value: PageViewCount ->
            ps.setString(1, value.page)
            ps.setTimestamp(2, java.sql.Timestamp(value.windowStart))
            ps.setTimestamp(3, java.sql.Timestamp(value.windowEnd))
            ps.setLong(4, value.count)
        },
        JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
            .withUrl("jdbc:mysql://mysql:3306/stats")
            .withDriverName("com.mysql.cj.jdbc.Driver")
            .withUsername("user")
            .withPassword("pass")
            .build()
    )

    windowed.addSink(jdbcSink)

    env.execute("Page Views Streaming Job (DataStream API)")
}
```

**Key points in the DataStream version:**

- **WatermarkStrategy:** Assigns timestamps and allows late events up to 5 seconds.
- **KeyBy and window:** Standard Flink operators. In Kotlin, we can use lambda for key selector.
- **Window function:** A simple `apply` that counts events in the window. Flink also provides `aggregate`, `reduce`, etc.
- **JDBC sink:** Uses Flink's `JdbcSink` with a prepared statement setter.

**Idiomatic Kotlin:** We use Kotlin's `apply` for property initialization, data classes for POJOs, and lambda syntax where appropriate (e.g., key selector). The `Collector` in the window function is from Flink's Java API; we use a SAM conversion for the `WindowFunction` interface.

---

## Summary

Both examples demonstrate how to leverage Kotlin's concise syntax while using the powerful Java APIs of Spark and Flink. The key is to:

- Use data classes for type-safe records.
- Use lambdas for simple functions (e.g., key selectors, map operations).
- Take advantage of Kotlin's `apply`, `let`, and other scoping functions for cleaner configuration.
- Remember that all Java APIs are directly callable.

These examples should give you a solid foundation to discuss distributed data processing in Kotlin during your interview. Good luck!
