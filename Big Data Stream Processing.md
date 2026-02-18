# 📘 Complete Big Data & Stream Processing Guide for ML Interviews

This guide covers **everything** you need to know about big data processing, stream processing, Apache Spark, Apache Flink – with **detailed theory**, **idiomatic Kotlin code examples**, **explicit input data**, **expected outputs**, **performance pitfalls**, and **optimizations**. It’s your ultimate cheat sheet to crack any data-focused ML interview.

---

## Part 1: Big Data Processing – Core Concepts

### 1.1 What Is Big Data?
Big data refers to datasets too large or complex for traditional single-node systems. Characterized by the **3 Vs**:
- **Volume**: Terabytes to petabytes
- **Velocity**: High speed of data generation (streaming)
- **Variety**: Structured, semi‑structured, unstructured

### 1.2 Distributed Computing Principles
- **Data Locality**: Move computation to data, not vice versa (minimizes network I/O)
- **Partitioning**: Split data across nodes; each node processes its partition in parallel
- **Replication**: Copy data for fault tolerance
- **Consistency Models**: Strong vs. eventual consistency
- **MapReduce**: Original paradigm (map → shuffle → reduce); Spark improved on this with in‑memory DAGs

### 1.3 Batch vs. Stream Processing
| | **Batch Processing** | **Stream Processing** |
|---|---|---|
| **Data** | Finite, bounded | Infinite, unbounded |
| **Latency** | Minutes to hours | Milliseconds to seconds |
| **Examples** | Daily ETL, model training | Fraud detection, real‑time dashboards |
| **Tools** | Spark, Hive | Flink, Kafka Streams, Spark Streaming |

### 1.4 Key Concepts in Distributed Data Processing
- **Parallelism**: Number of concurrent tasks (partitions, slots, cores)
- **Fault Tolerance**: Recover from failures without data loss (checkpoints, lineage)
- **Exactly‑Once Semantics**: Each record processed exactly once (no duplicates, no misses)
- **Data Skew**: Uneven distribution of data across partitions → performance bottleneck
- **Shuffle**: Redistributing data across partitions (expensive – network I/O, disk)

### 1.5 Storage Formats for Big Data
| Format | Type | Best For | Compression | Predicate Pushdown |
|--------|------|----------|-------------|-------------------|
| **Parquet** | Columnar | Analytics, ML features | ✅ Snappy/Zstd | ✅ |
| **ORC** | Columnar | Hive integration | ✅ Zlib | ✅ |
| **Avro** | Row | Write‑heavy, schema evolution | ✅ | ❌ |
| **JSON** | Row | Human‑readable, ingest | ❌ | ❌ |
| **CSV** | Row | Legacy systems | ❌ | ❌ |

**Rule for ML**: Always use **Parquet** for training data – columnar storage speeds up feature selection and reduces I/O.

---

## Part 2: Stream Processing – Deeper Theory

### 2.1 Event Time vs. Processing Time
- **Event Time**: When the event actually occurred (embedded in data)
- **Processing Time**: When the system processes the event

Stream processing engines must handle **out‑of‑order events** and **late data**. This is where **watermarks** come in.

### 2.2 Watermarks
A watermark is a threshold that indicates “no events with timestamp ≤ watermark will arrive”. Example: watermark = maxEventTime – 5 seconds means we tolerate up to 5 seconds of lateness.

### 2.3 Windowing
Group events into finite chunks for aggregation:
- **Tumbling windows**: Fixed size, non‑overlapping
- **Sliding windows**: Fixed size, overlapping (every slide interval)
- **Session windows**: Dynamic, based on gaps of inactivity

### 2.4 State Management
Operators often need to remember past events (e.g., counts, aggregates). State is stored locally on the task manager, and periodically checkpointed to durable storage for recovery.

### 2.5 Backpressure
When a downstream operator is slower than upstream, backpressure signals upstream to slow down, preventing system overload.

### 2.6 Exactly‑Once Semantics in Stream Processing
Achieved via:
- **Idempotent writes** + **transactional sinks** (e.g., Kafka transactional producer)
- **Distributed snapshots** (Flink’s checkpoints): periodically save state and offsets; on failure, restore from last checkpoint and replay from saved offsets.

---

## Part 3: Apache Spark – Complete Guide with Input/Output

### 3.1 Spark Architecture

```
Client → Driver (SparkContext) → DAG Scheduler → Task Scheduler → Executors
         ↑                      ↓
    Cluster Manager        Task Queues
    (YARN/K8s)
```

- **Driver**: Runs main(), creates SparkContext, schedules tasks
- **Executors**: Worker processes that run tasks and cache data
- **Cluster Manager**: Allocates resources (YARN, Kubernetes, standalone)

### 3.2 Core Abstractions
| Abstraction | Description | When to Use |
|-------------|-------------|-------------|
| **RDD** | Low‑level, type‑safe, functional | Custom transformations, no schema |
| **DataFrame** | Distributed rows with schema, Catalyst optimised | Most ETL, SQL, ML features |
| **Dataset** | Type‑safe version of DataFrame (JVM only) | When type safety is critical |

**Lazy Evaluation**: Transformations are lazy; actions (like `show()`, `count()`) trigger DAG execution.

**Catalyst Optimizer**: Query optimizer – analysis → logical optimisation → physical planning → code generation.

**Tungsten**: Off‑heap memory management and code generation for speed.

### 3.3 Idiomatic Kotlin with Spark

We’ll use the **Kotlin for Apache Spark** API (by JetBrains). Add dependency:

```kotlin
// build.gradle.kts
dependencies {
    implementation("org.jetbrains.kotlin:kotlin-spark-api:1.2.3")
    implementation("org.apache.spark:spark-sql_2.12:3.3.0")
}
```

#### Example 1: Basic ETL with Explicit Input and Output

**Input Data (`events.json`)**
```json
{"userId": "u123", "eventType": "purchase", "timestamp": 1609459200000, "value": 250.0}
{"userId": "u456", "eventType": "purchase", "timestamp": 1609459201000, "value": 75.5}
{"userId": "u123", "eventType": "click", "timestamp": 1609459205000, "value": 0.0}
{"userId": "u789", "eventType": "purchase", "timestamp": 1609459210000, "value": 125.0}
{"userId": "u456", "eventType": "purchase", "timestamp": 1609459215000, "value": 200.0}
{"userId": "u123", "eventType": "purchase", "timestamp": 1609459220000, "value": 300.0}
```

**Kotlin Code**
```kotlin
import org.jetbrains.spark.api.*

fun main() = withSpark {
    val df = spark.read().json("events.json")

    println("=== Schema ===")
    df.printSchema()

    println("=== Raw Data (first 5 rows) ===")
    df.show(5, truncate = false)

    val result = df
        .filter(col("eventType") === "purchase")
        .groupBy("userId")
        .agg(
            sum("value") as "total_spent",
            count("*") as "purchase_count"
        )
        .orderBy(col("total_spent").desc())

    println("=== Aggregated Result ===")
    result.show(truncate = false)

    result.write().parquet("output/purchase_summary")
}
```

**Expected Output**
```
=== Schema ===
root
 |-- eventType: string (nullable = true)
 |-- timestamp: long (nullable = true)
 |-- userId: string (nullable = true)
 |-- value: double (nullable = true)

=== Raw Data (first 5 rows) ===
+---------+-------------+------+-----+
|eventType|timestamp    |userId|value|
+---------+-------------+------+-----+
|purchase |1609459200000|u123  |250.0|
|purchase |1609459201000|u456  |75.5 |
|click    |1609459205000|u123  |0.0  |
|purchase |1609459210000|u789  |125.0|
|purchase |1609459215000|u456  |200.0|
+---------+-------------+------+-----+

=== Aggregated Result ===
+------+-----------+--------------+
|userId|total_spent|purchase_count|
+------+-----------+--------------+
|u123  |550.0      |2             |
|u456  |275.5      |2             |
|u789  |125.0      |1             |
+------+-----------+--------------+
```

---

#### Example 2: Join Optimisation – Broadcast Join

**Input Data**

**transactions.csv**
```
transactionId,userId,amount,timestamp
t1,u123,250.0,1609459200000
t2,u456,75.5,1609459201000
t3,u123,300.0,1609459220000
t4,u789,125.0,1609459210000
t5,u456,200.0,1609459215000
... (billions more)
```

**users.csv**
```
userId,name,city
u123,Alice,NYC
u456,Bob,LA
u789,Charlie,Chicago
```

**Kotlin Code**
```kotlin
import org.apache.spark.sql.functions.broadcast

fun joinExample(spark: SparkSession) {
    val transactions = spark.read().option("header", true).csv("transactions.csv")
        .withColumn("amount", col("amount").cast("double"))
    val users = spark.read().option("header", true).csv("users.csv")

    // Slow join
    val badJoin = transactions.join(users, "userId")
    println("=== Slow Join Physical Plan ===")
    badJoin.explain("extended")  // shows SortMergeJoin

    // Optimised broadcast join
    val goodJoin = transactions.join(broadcast(users), "userId")
    println("=== Optimised Join Physical Plan ===")
    goodJoin.explain("extended") // shows BroadcastHashJoin

    println("=== Result (first 5 rows) ===")
    goodJoin.select("userId", "name", "city", "amount").show(5, truncate = false)
}
```

**Expected Output (physical plans truncated)**
```
=== Slow Join Physical Plan ===
== Physical Plan ==
*(3) SortMergeJoin [userId], [userId], Inner
:- *(1) Sort [userId ASC], false, 0
:  +- *(1) Project [userId, amount, timestamp]
:     +- *(1) FileScan csv [userId, amount, timestamp] ...
+- *(2) Sort [userId ASC], false, 0
   +- *(2) FileScan csv [userId, name, city] ...

=== Optimised Join Physical Plan ===
== Physical Plan ==
*(1) BroadcastHashJoin [userId], [userId], Inner, BuildRight
:- *(1) Project [userId, amount, timestamp]
:  +- *(1) FileScan csv [userId, amount, timestamp] ...
+- BroadcastExchange HashedRelation
   +- *(1) FileScan csv [userId, name, city] ...

=== Result (first 5 rows) ===
+------+-----+----+------+
|userId|name |city|amount|
+------+-----+----+------+
|u123  |Alice|NYC |250.0 |
|u123  |Alice|NYC |300.0 |
|u456  |Bob  |LA  |75.5  |
|u456  |Bob  |LA  |200.0 |
|u789  |Charlie|Chicago|125.0|
+------+-----+----+------+
```

---

#### Example 3: Window Functions – Slow vs. Fast

**Input Data (`user_activity.csv`)**
```
userId,timestamp,amount
u123,1609459200000,100.0
u123,1609545600000,50.0
u123,1609632000000,75.0
u456,1609459200000,200.0
u456,1609545600000,150.0
u456,1609632000000,125.0
```

**Kotlin Code**
```kotlin
import org.apache.spark.sql.expressions.Window

fun windowExample(spark: SparkSession) {
    val df = spark.read().option("header", true).csv("user_activity.csv")
        .withColumn("timestamp", col("timestamp").cast("long"))
        .withColumn("amount", col("amount").cast("double"))

    // Slow global window
    val globalWindow = Window.orderBy("timestamp")
    val slowDf = df.withColumn("global_rank", rank().over(globalWindow))
    println("=== Slow Window Physical Plan ===")
    slowDf.explain("extended")

    // Fast partitioned window
    val perUserWindow = Window
        .partitionBy("userId")
        .orderBy("timestamp")
        .rowsBetween(Window.unboundedPreceding(), Window.currentRow())

    val fastDf = df.withColumn("user_cumulative_sum", sum("amount").over(perUserWindow))

    println("=== Fast Window Result ===")
    fastDf.select("userId", "timestamp", "amount", "user_cumulative_sum").show(truncate = false)
}
```

**Expected Output**
```
=== Slow Window Physical Plan ===
...
== Physical Plan ==
Window [rank(timestamp) windowspecdefinition(timestamp ASC, ...)], [timestamp ASC]
+- *(1) Sort [timestamp ASC], false, 0
   +- Exchange SinglePartition  <-- All data to one partition!
      ...

=== Fast Window Result ===
+------+-------------+------+-------------------+
|userId|timestamp    |amount|user_cumulative_sum|
+------+-------------+------+-------------------+
|u123  |1609459200000|100.0 |100.0              |
|u123  |1609545600000|50.0  |150.0              |
|u123  |1609632000000|75.0  |225.0              |
|u456  |1609459200000|200.0 |200.0              |
|u456  |1609545600000|150.0 |350.0              |
|u456  |1609632000000|125.0 |475.0              |
+------+-------------+------+-------------------+
```

---

#### Example 4: Handling Data Skew with Salting

**Input Data**

**transactions.csv** (skewed on `country`)
```
transactionId,userId,country,amount
t1,u123,US,100.0
t2,u456,US,200.0
t3,u789,US,150.0
t4,u234,US,300.0
t5,u567,US,250.0
... 90% of rows are "US"
t100,u999,CA,50.0
t101,u111,UK,75.0
```

**countries.csv**
```
country,region,continent
US,North America,Americas
CA,North America,Americas
UK,Western Europe,Europe
```

**Kotlin Code**
```kotlin
fun fixSkewJoin(transactions: DataFrame, countries: DataFrame): DataFrame {
    val saltedTransactions = transactions
        .withColumn("salt", (rand() * 10).cast("int"))
        .withColumn("salted_key", concat(col("country"), lit("_"), col("salt")))

    val saltedCountries = countries
        .withColumn("salt", explode(array(0,1,2,3,4,5,6,7,8,9)))
        .withColumn("salted_key", concat(col("country"), lit("_"), col("salt")))

    return saltedTransactions
        .join(saltedCountries, "salted_key")
        .drop("salt", "salted_key")
}

fun main() = withSpark {
    val transactions = spark.read().option("header", true).csv("transactions.csv")
        .withColumn("amount", col("amount").cast("double"))
    val countries = spark.read().option("header", true).csv("countries.csv")

    val result = fixSkewJoin(transactions, countries)
    result.select("transactionId", "userId", "country", "region", "amount").show(10, truncate = false)
}
```

**Expected Output (first 10 rows)**
```
+-------------+------+-------+-------------+------+
|transactionId|userId|country|region       |amount|
+-------------+------+-------+-------------+------+
|t1           |u123  |US     |North America|100.0 |
|t2           |u456  |US     |North America|200.0 |
|t3           |u789  |US     |North America|150.0 |
|t4           |u234  |US     |North America|300.0 |
|t5           |u567  |US     |North America|250.0 |
|...          |...   |US     |North America|...   | (US rows spread across partitions)
|t100         |u999  |CA     |North America|50.0  |
|t101         |u111  |UK     |Western Europe|75.0 |
+-------------+------+-------+-------------+------+
```

---

#### Example 5: ML Pipeline with Caching

**Input Data (`user_features.parquet`)**

| userId | feature1 | feature2 | category | label |
|--------|----------|----------|----------|-------|
| u1     | 0.5      | 1.2      | A        | 1     |
| u2     | 1.3      | 0.8      | B        | 0     |
| u3     | 2.1      | 3.4      | A        | 1     |
| ...    | ...      | ...      | ...      | ...   |

**Kotlin Code**
```kotlin
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline

fun mlPipeline(spark: SparkSession) {
    val data = spark.read().parquet("user_features.parquet")
    val Array(train, test) = data.randomSplit(doubleArrayOf(0.8, 0.2), seed = 42)

    val stringIndexer = StringIndexer()
        .setInputCol("category")
        .setOutputCol("categoryIndex")
        .setHandleInvalid("keep")

    val assembler = VectorAssembler()
        .setInputCols(arrayOf("feature1", "feature2", "categoryIndex"))
        .setOutputCol("features")

    val rf = RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setNumTrees(20)

    val pipeline = Pipeline().setStages(array(stringIndexer, assembler, rf))

    // Cache training and test data
    train.cache()
    test.cache()

    val model = pipeline.fit(train)
    val predictions = model.transform(test)

    predictions.select("label", "prediction").show(10, truncate = false)

    train.unpersist()
    test.unpersist()
}
```

**Expected Output**
```
+-----+----------+
|label|prediction|
+-----+----------+
|1    |1.0       |
|0    |0.0       |
|1    |1.0       |
|0    |0.0       |
|1    |1.0       |
|0    |1.0       |   (misclassified)
|...  |...       |
+-----+----------+
```

---

### 3.4 Spark Performance Tuning Cheat Sheet

| Parameter | Recommendation | Why |
|-----------|---------------|-----|
| `spark.sql.adaptive.enabled=true` | Always enable | Dynamic coalescing, skew join handling |
| `spark.sql.adaptive.skewJoin.enabled=true` | Enable | Handles skewed keys automatically |
| `spark.sql.shuffle.partitions` | 200‑500 | Increase for large data, decrease for small |
| `spark.executor.memory` | 4‑8GB per executor | Avoid >64GB (GC overhead) |
| `spark.memory.fraction` | 0.6‑0.7 | Fraction for execution+storage |
| `spark.serializer` | `KryoSerializer` | 10x faster than Java serialisation |
| File format | Parquet/ORC | Columnar, compression, predicate pushdown |
| `spark.sql.autoBroadcastJoinThreshold` | 10MB (default) | Increase if you have memory, but be careful |

---

## Part 4: Apache Flink – Complete Guide with Input/Output

### 4.1 Flink Architecture

```
Client → JobManager (Active/Standby) → TaskManagers
         ↑            ↓
    ZooKeeper    Checkpoints (HDFS/S3)
    (HA mode)
```

- **JobManager**: Coordinates checkpoints, recovery, schedules tasks
- **TaskManager**: Worker nodes with multiple task slots
- **State Backend**: Where state lives (heap or RocksDB)
- **Checkpoints**: Periodic snapshots of state + source offsets for exactly‑once recovery

### 4.2 Core Concepts

| Concept | Explanation |
|---------|-------------|
| **Stream** | Unbounded data flow |
| **Transformation** | Map, filter, keyBy, window, etc. |
| **Operator** | Parallel instance of a transformation |
| **Parallelism** | Number of parallel subtasks per operator |
| **State** | Local data (counters, aggregates) |
| **Checkpoint** | Snapshot of state + offsets to durable storage |
| **Watermark** | Timestamp tracker for event time |
| **Window** | Group events into finite chunks |

### 4.3 Idiomatic Kotlin with Flink

Flink doesn’t have an official Kotlin API, but Java API works perfectly. Add dependencies:

```kotlin
implementation("org.apache.flink:flink-java:1.17.0")
implementation("org.apache.flink:flink-streaming-java:1.17.0")
implementation("org.apache.flink:flink-table-api-java-bridge:1.17.0")
implementation("org.apache.flink:flink-connector-kafka:1.17.0")
```

#### Example 1: Basic Stream Processing with Event Time

**Input Data (simulated stream)**
```kotlin
val clicks = listOf(
    ClickEvent("u1", "home", 1609459200000L),
    ClickEvent("u2", "products", 1609459201000L),
    ClickEvent("u1", "checkout", 1609459205000L),
    ClickEvent("u3", "home", 1609459210000L),
    ClickEvent("u2", "home", 1609459215000L),        // on time
    ClickEvent("u1", "home", 1609459190000L)         // 10 seconds early (late for previous window)
)
```

**Kotlin Code**
```kotlin
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
import org.apache.flink.api.common.eventtime.WatermarkStrategy
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows
import org.apache.flink.util.Collector
import java.time.Duration

data class ClickEvent(val userId: String, val page: String, val timestamp: Long)

fun main() {
    val env = StreamExecutionEnvironment.getExecutionEnvironment()
    env.setParallelism(2)

    val clicks = env.fromCollection(
        listOf(
            ClickEvent("u1", "home", 1609459200000L),
            ClickEvent("u2", "products", 1609459201000L),
            ClickEvent("u1", "checkout", 1609459205000L),
            ClickEvent("u3", "home", 1609459210000L),
            ClickEvent("u2", "home", 1609459215000L),
            ClickEvent("u1", "home", 1609459190000L)  // late
        )
    )

    val withTimestamps = clicks.assignTimestampsAndWatermarks(
        WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(10))
            .withTimestampAssigner { event, _ -> event.timestamp }
    )

    val result = withTimestamps
        .keyBy { it.page }
        .window(TumblingEventTimeWindows.of(Time.minutes(5)))
        .apply { page, window, events, out: Collector<String> ->
            out.collect("Page '$page' had ${events.count()} visits in window ${window.start} to ${window.end}")
        }

    result.print()
    env.execute("Click Analytics")
}
```

**Expected Output**
```
Page 'home' had 2 visits in window 1609459200000 to 1609459500000
Page 'products' had 1 visits in window 1609459200000 to 1609459500000
Page 'checkout' had 1 visits in window 1609459200000 to 1609459500000
Page 'home' had 1 visits in window 1609458900000 to 1609459200000   (late event goes to previous window)
```

---

#### Example 2: Stateful Processing – Fraud Detection

**Input Data (stream)**
```
Transaction(userId=u123, amount=50.0, timestamp=1609459200000)
Transaction(userId=u123, amount=300.0, timestamp=1609459201000)   // 1 second later, 6x amount
Transaction(userId=u456, amount=20.0, timestamp=1609459200000)
Transaction(userId=u456, amount=25.0, timestamp=1609459205000)   // 5 seconds later, small increase
```

**Kotlin Code**
```kotlin
import org.apache.flink.api.common.state.{ValueState, ValueStateDescriptor}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.KeyedProcessFunction
import org.apache.flink.util.Collector

data class Transaction(val userId: String, val amount: Double, val timestamp: Long)
data class Alert(val userId: String, val reason: String)

class FraudDetector : KeyedProcessFunction<String, Transaction, Alert>() {
    
    @transient private var lastTransactionState: ValueState<Transaction>? = null

    override fun open(parameters: Configuration) {
        val desc = ValueStateDescriptor("last-transaction", Transaction::class.java)
        lastTransactionState = runtimeContext.getState(desc)
    }

    override fun processElement(
        transaction: Transaction,
        ctx: Context,
        out: Collector<Alert>
    ) {
        val last = lastTransactionState?.value()

        if (last != null) {
            val timeDiff = transaction.timestamp - last.timestamp
            val amountRatio = transaction.amount / last.amount

            if (timeDiff < 60000 && amountRatio > 5.0) {
                out.collect(Alert(transaction.userId, "Suspicious: large amount within short time"))
            }
        }

        lastTransactionState?.update(transaction)
    }
}

fun main() {
    val env = StreamExecutionEnvironment.getExecutionEnvironment()
    val transactions = env.fromElements(
        Transaction("u123", 50.0, 1609459200000L),
        Transaction("u123", 300.0, 1609459201000L),
        Transaction("u456", 20.0, 1609459200000L),
        Transaction("u456", 25.0, 1609459205000L)
    )

    val alerts = transactions
        .keyBy { it.userId }
        .process(FraudDetector())

    alerts.print()
    env.execute("Fraud Detection")
}
```

**Expected Output**
```
Alert(userId=u123, reason=Suspicious: large amount within short time)
```

---

#### Example 3: Flink SQL with Windowing

**Input Data (Kafka topic `clicks`)**
```json
{"userId": "u1", "page": "home", "clickTime": "2025-12-08 10:00:00"}
{"userId": "u2", "page": "products", "clickTime": "2025-12-08 10:01:00"}
{"userId": "u1", "page": "checkout", "clickTime": "2025-12-08 10:02:00"}
{"userId": "u3", "page": "home", "clickTime": "2025-12-08 10:03:00"}
{"userId": "u2", "page": "home", "clickTime": "2025-12-08 10:04:00"}
```

**Kotlin Code**
```kotlin
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment

fun flinkSqlExample(tableEnv: StreamTableEnvironment) {
    tableEnv.executeSql("""
        CREATE TABLE clicks (
            userId STRING,
            page STRING,
            clickTime TIMESTAMP(3),
            WATERMARK FOR clickTime AS clickTime - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'clicks',
            'properties.bootstrap.servers' = 'localhost:9092',
            'format' = 'json'
        )
    """)

    tableEnv.executeSql("""
        CREATE TABLE page_stats (
            page STRING,
            window_start TIMESTAMP(3),
            window_end TIMESTAMP(3),
            click_count BIGINT
        ) WITH (
            'connector' = 'print'
        )
    """)

    val result = tableEnv.sqlQuery("""
        SELECT 
            page,
            TUMBLE_START(clickTime, INTERVAL '10' MINUTE) AS window_start,
            TUMBLE_END(clickTime, INTERVAL '10' MINUTE) AS window_end,
            COUNT(*) AS click_count
        FROM clicks
        GROUP BY page, TUMBLE(clickTime, INTERVAL '10' MINUTE)
    """)

    result.executeInsert("page_stats")
}
```

**Expected Output (printed to stdout)**
```
+I[home, 2025-12-08T10:00, 2025-12-08T10:10, 2]
+I[products, 2025-12-08T10:00, 2025-12-08T10:10, 1]
+I[checkout, 2025-12-08T10:00, 2025-12-08T10:10, 1]
```

---

### 4.4 Flink Performance Tuning

| Area | Parameter | Recommendation |
|------|-----------|---------------|
| **State Backend** | `state.backend` | `rocksdb` for large state (>1GB), `hashmap` for small |
| **RocksDB** | `state.backend.rocksdb.localdir` | Point to SSD for performance |
| **Checkpointing** | `execution.checkpointing.interval` | 1‑5 minutes for production |
| | `execution.checkpointing.timeout` | 10 minutes (avoid hanging) |
| **Network Buffers** | `taskmanager.network.memory.fraction` | 0.1‑0.2 for shuffle‑heavy jobs |
| **Parallelism** | `parallelism.default` | Start with CPU cores * 1.5 |
| **Idle Sources** | `table.exec.source.idle-timeout` | 1 second (release resources) |

---

## Part 5: Spark vs. Flink – Decision Guide

| Requirement | Recommendation |
|-------------|----------------|
| Need millisecond latency? | **Flink** (true streaming) |
| Very large state (>100GB)? | **Flink** (RocksDB backend) |
| Complex event patterns (CEP)? | **Flink** (built‑in CEP library) |
| Already using Spark ecosystem? | **Spark Streaming** (unified APIs) |
| Batch + streaming unified? | Both (Spark has unified DataFrame API) |
| Python‑only team? | **Spark** (PySpark mature) |
| Exactly‑once with minimal overhead? | **Flink** (lightweight checkpoints) |
| Ease of debugging? | **Spark** (micro‑batch = easier reasoning) |

---

## Part 6: Common Interview Q&A

**Q: Why did Uber move from Spark to Flink for ingestion?**  
**A:** Three reasons: 1) Data freshness from hours to minutes, 2) 25% compute cost reduction (no wasted batch scheduling), 3) Better handling of streaming‑specific challenges like watermarks and large state.

**Q: How does Flink achieve exactly‑once semantics?**  
**A:** Through **asynchronous distributed snapshots (checkpoints)**. Flink periodically saves operator state and source offsets. On failure, sources rewind, state restored, and processing resumes without duplicates.

**Q: When does data skew occur and how to fix it?**  
**A:** When one key dominates (e.g., 90% data for "US"). Fixes: salting for joins, repartitioning, broadcast joins for small tables, or AQE skew join in Spark 3+.

**Q: Parquet vs ORC – which is better for ML?**  
**A:** Parquet has better Spark integration and wider adoption; both are columnar. For ML, Parquet is preferred.

**Q: How do you debug a slow Spark job?**  
**A:** 1) Check Spark UI – stages with long tasks indicate skew; 2) Look at shuffle spill (memory pressure); 3) Verify data partitioning; 4) Check if broadcasting can help; 5) Enable AQE.

**Q: What is a watermark in Flink?**  
**A:** A watermark is a metric that tracks event time progress and signals that no events with timestamp ≤ watermark will arrive. Used to handle out‑of‑order and late data.

---

## Part 7: Performance Optimization Summary (Cheat Sheet)

| Problem | Spark Solution | Flink Solution |
|---------|---------------|----------------|
| **Data Skew** | Salting, AQE | Rebalance(), custom partitioner |
| **Small Files** | coalesce(), repartition() | Row‑group merging, compaction |
| **Slow Joins** | Broadcast if possible | Broadcast in Table API |
| **Memory Pressure** | Increase partitions, Kryo | RocksDB backend |
| **GC Overhead** | Tune `spark.memory.fraction` | Off‑heap state (RocksDB) |
| **Backpressure** | Not automatic (micro‑batch) | Built‑in, auto‑throttling |

---

_____________________________
# Handons (Coding examples)
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

This guide gives you both **deep theory** and **practical Kotlin code** with **explicit input data** and **expected output** for every example. Use it to build confidence and nail your interview. Good luck! 🚀
