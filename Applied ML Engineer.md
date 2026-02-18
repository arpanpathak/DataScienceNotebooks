# Apple ML Platform Engineer Interview: Complete Study Guide

## Table of Contents
1. [Kubernetes & Container Orchestration (Deep Dive)](#1-kubernetes--container-orchestration-deep-dive)
   - Architecture
   - Core Objects with Examples
   - Autoscaling (HPA, VPA, KEDA)
   - Multi-tenancy & Resource Management
   - Networking & Storage
2. [Distributed Systems & Big Data](#2-distributed-systems--big-data)
   - Apache Spark (with code examples & internals)
   - Apache Flink (concepts + PyFlink example)
   - Ray (tasks, actors + ML examples)
   - Distributed System Principles
3. [MLOps & ML Lifecycle](#3-mlops--ml-lifecycle)
   - MLflow (tracking, registry, serving)
   - CI/CD for ML (vs traditional CI/CD)
   - Model Serving Patterns
4. [Reliability Engineering & Performance](#4-reliability-engineering--performance)
   - SRE Principles (SLIs, SLOs, error budgets)
   - Performance Analysis (USE, RED methods)
   - Troubleshooting Frameworks
5. [Interview Question Templates & Scenarios](#5-interview-question-templates--scenarios)
   - How to structure answers (STAR)
   - Common scenario questions with walkthroughs
6. [Quick Reference Cheat Sheet](#6-quick-reference-cheat-sheet)

---

## 1. Kubernetes & Container Orchestration (Deep Dive)

### Why Kubernetes for ML Platforms?
ML workloads are diverse: batch training jobs, real-time inference, data preprocessing, hyperparameter tuning. Kubernetes provides a unified control plane to schedule these workloads efficiently across a cluster of machines. It handles resource isolation (via containers), auto-scaling, self-healing, and rolling updates. For an ML platform team, Kubernetes is the foundation.

### Architecture Explained

**Control Plane (Master):**
- **kube-apiserver:** The front door. All `kubectl` commands, REST calls, and internal components communicate here. It validates and configures data for the API objects.
- **etcd:** Consistent and highly-available key-value store. Stores all cluster data (state, configuration). Think of it as the cluster's "source of truth." If etcd goes down, the cluster cannot recover.
- **kube-scheduler:** Watches for newly created pods with no assigned node, then selects a node for them to run on based on resource requirements, policies, affinity/anti-affinity specs, and data locality.
- **kube-controller-manager:** Runs controllers that regulate the state of the cluster. Examples: Node controller (notices node failures), Replication controller (maintains correct number of pods), Endpoint controller (updates Service endpoints).

**Worker Nodes:**
- **kubelet:** An agent that runs on each node. It ensures containers are running in a pod as expected. It communicates with the container runtime (e.g., containerd) to start/stop containers.
- **kube-proxy:** Maintains network rules on nodes. It enables communication to your pods from inside or outside the cluster.
- **Container Runtime:** The software that actually runs containers (containerd, CRI-O, Docker). Kubernetes uses the Container Runtime Interface (CRI) to interact with it.

### Core Objects with Examples

#### Pods
The smallest deployable unit. A pod encapsulates one or more containers, storage resources, a unique network IP, and options that govern how the container(s) should run.

**Example: Simple Pod**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference
  labels:
    app: sentiment-model
    version: v1
spec:
  containers:
  - name: inference-engine
    image: myregistry/sentiment:latest
    ports:
    - containerPort: 8080
    resources:
      requests:
        memory: "512Mi"
        cpu: "250m"
      limits:
        memory: "1Gi"
        cpu: "500m"
    env:
    - name: MODEL_PATH
      value: "/models/bert-base"
    volumeMounts:
    - name: model-storage
      mountPath: "/models"
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-pvc
```
**Explanation:**
- `metadata.labels` are key/value pairs used to organize and select subsets of objects.
- `spec.containers.resources` defines the minimum guaranteed resources (`requests`) and maximum allowed (`limits`). This is crucial for scheduling and avoiding resource starvation.
- `env` sets environment variables inside the container.
- `volumeMounts` attaches a volume to the container at a specific path.
- `volumes` defines the volume source—here a PersistentVolumeClaim (PVC) that requests persistent storage.

#### Init Containers
Specialized containers that run to completion before the main containers start. Useful for setup tasks like downloading models, waiting for dependencies, or initializing data.

**Example: Init Container to Download a Large Model**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-inference-with-init
spec:
  initContainers:
  - name: model-downloader
    image: curlimages/curl:latest
    command: ['curl', '-L', 'https://storage.googleapis.com/models/latest.bin', '-o', '/models/model.bin']
    volumeMounts:
    - name: model-storage
      mountPath: /models
  containers:
  - name: inference-server
    image: myapp/inference:latest
    volumeMounts:
    - name: model-storage
      mountPath: /models
  volumes:
  - name: model-storage
    emptyDir: {}
```
**Explanation:**
- `initContainers` run sequentially; each must exit successfully before the next starts.
- After the init container downloads the model to `/models/model.bin`, the main container can access it from the same `emptyDir` volume (a temporary directory that lives as long as the pod). The `emptyDir` is initially empty, but the init container fills it.
- This pattern avoids embedding the model in the container image, reducing image size and allowing model updates without rebuilding the image.

#### Deployments vs. StatefulSets

**Deployment:** Manages a set of identical pods. Suitable for stateless applications where any pod can handle any request. Provides rolling updates and rollback.

**StatefulSet:** Manages stateful applications that require stable network identities and persistent storage. Each pod gets a unique ordinal index (e.g., `db-0`, `db-1`) and persistent volumes that persist across rescheduling.

**When to use StatefulSet in ML:**
- Distributed training with parameter servers (each worker needs a stable identity).
- Databases like MySQL, Cassandra, or Kafka.
- Services where each instance has its own storage (e.g., feature store nodes).

**StatefulSet Example (MongoDB cluster):**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: mongodb
spec:
  clusterIP: None  # Headless service for stable network identities
  selector:
    app: mongodb
  ports:
  - port: 27017
    name: mongodb

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
spec:
  serviceName: "mongodb"
  replicas: 3
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
      - name: mongodb
        image: mongo:4.4
        ports:
        - containerPort: 27017
        volumeMounts:
        - name: data
          mountPath: /data/db
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```
**Explanation:**
- `clusterIP: None` creates a headless service, allowing direct pod-to-pod DNS resolution (e.g., `mongodb-0.mongodb.default.svc.cluster.local`).
- `volumeClaimTemplates` automatically creates a PVC for each replica (data-mongodb-0, data-mongodb-1, data-mongodb-2). If a pod is rescheduled, it reattaches to the same PVC.
- This setup allows the MongoDB cluster to form a replica set with stable members.

#### ConfigMaps & Secrets

**ConfigMap:** Stores non-confidential configuration data in key-value pairs. Pods can consume ConfigMaps as environment variables, command-line arguments, or files in a volume.

**Secret:** Similar to ConfigMap but intended for sensitive data (base64 encoded). Can be mounted as files or exposed as environment variables.

**Why separate them?** Security best practices: don't hardcode credentials in images or pod specs. Secrets are encrypted at rest in etcd (if configured) and can be managed with RBAC.

**Example: ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
data:
  model_threshold: "0.85"
  batch_size: "32"
  log_level: "INFO"
```

**Example: Secret (base64 encoded)**
```bash
echo -n 'my-api-key' | base64
# Output: bXktYXBpLWtleQ==
```
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
type: Opaque
data:
  api-key: bXktYXBpLWtleQ==
```

**Pod consuming both:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-app
spec:
  containers:
  - name: app
    image: myapp/ml:latest
    env:
    - name: THRESHOLD
      valueFrom:
        configMapKeyRef:
          name: ml-config
          key: model_threshold
    - name: API_KEY
      valueFrom:
        secretKeyRef:
          name: api-credentials
          key: api-key
    - name: BATCH_SIZE
      valueFrom:
        configMapKeyRef:
          name: ml-config
          key: batch_size
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
  volumes:
  - name: config-volume
    configMap:
      name: ml-config
```
**Explanation:**
- `valueFrom` injects specific keys as environment variables.
- The volume mount with `configMap` creates a file for each key in `/etc/config` (e.g., `/etc/config/model_threshold` containing "0.85").

### Autoscaling for ML Workloads

#### HPA (Horizontal Pod Autoscaler)
Scales the number of pod replicas based on observed CPU/memory utilization or custom metrics.

**How it works:** The HPA controller periodically queries metrics (via Metrics Server or custom metrics API), calculates the desired replicas as `desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]`, and scales the target resource (Deployment, StatefulSet).

**Example: CPU-based HPA**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
**Explanation:**
- When average CPU utilization across pods exceeds 70%, HPA increases replicas (up to 10). When below, it decreases (down to 2).

**Limitations for ML:** CPU/memory may not be the best scaling metric for inference services. You might want to scale based on request latency, queue depth, or GPU utilization. That's where custom metrics and KEDA come in.

#### KEDA (Kubernetes Event-Driven Autoscaler)
KEDA extends HPA by allowing scaling based on external event sources (Kafka, RabbitMQ, Prometheus, etc.). It can also scale to zero when no events are present—great for batch jobs.

**KEDA architecture:**
- **ScaledObject:** CRD that defines the scaling trigger and the target resource.
- **KEDA operator:** Watches ScaledObjects and creates an HPA that scales based on the external metric.

**Example: Scale inference pods based on GPU utilization (Prometheus query)**
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: gpu-inference-scaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpu-inference
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring.svc:9090
      metricName: gpu_utilization
      query: 'avg(DCGM_FI_DEV_GPU_UTIL) by (pod)'
      threshold: '80'
```
**Explanation:**
- KEDA queries Prometheus for GPU utilization per pod. If the average across pods exceeds 80%, KEDA signals the HPA to scale up. This ensures you have enough inference capacity when GPUs are busy.

#### VPA (Vertical Pod Autoscaler)
Automatically adjusts CPU and memory requests/limits for pods based on historical usage. Useful for right-sizing batch jobs where resource needs vary.

**Example VPA:**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: training-job-vpa
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: training-worker
  updatePolicy:
    updateMode: "Auto"  # or "Off" to only recommend
  resourcePolicy:
    containerPolicies:
    - containerName: '*'
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
      maxAllowed:
        cpu: "8"
        memory: "32Gi"
```
**Note:** VPA works best for workloads that can be restarted to apply new resource limits (e.g., batch jobs). For critical online services, you might use "Off" mode to get recommendations and apply them manually during maintenance.

### Multi-tenancy & Resource Management

In an ML platform, multiple teams share the same Kubernetes cluster. To ensure isolation and fair resource allocation, you need:

- **Namespaces:** Logical partitions within a cluster. Each team gets its own namespace.
- **ResourceQuotas:** Limit total resources (CPU, memory, storage, object counts) consumed by a namespace.
- **LimitRanges:** Set default min/max for pods/containers within a namespace.
- **Network Policies:** Control traffic between pods (e.g., isolate team namespaces).
- **PriorityClasses:** Ensure critical production jobs preempt less important experimental jobs.

**Example ResourceQuota (namespace `team-a`):**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-quota
  namespace: team-a
spec:
  hard:
    requests.cpu: "10"
    requests.memory: "20Gi"
    limits.cpu: "20"
    limits.memory: "40Gi"
    persistentvolumeclaims: "5"
    pods: "50"
    services: "10"
```
**Explanation:** Team A cannot collectively request more than 10 CPU cores and 20Gi memory. Their total usage cannot exceed 20 CPU cores and 40Gi memory. They can have at most 5 PVCs and 50 pods.

**Example LimitRange (per-container defaults):**
```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-container-limits
  namespace: team-a
spec:
  limits:
  - max:
      cpu: "4"
      memory: "8Gi"
    min:
      cpu: "50m"
      memory: "64Mi"
    default:
      cpu: "500m"
      memory: "1Gi"
    defaultRequest:
      cpu: "250m"
      memory: "512Mi"
    type: Container
```
**Explanation:** Any container created in namespace `team-a` without explicit requests/limits will get default request 250m CPU and 512Mi memory, and default limit 500m CPU and 1Gi memory. It also ensures containers don't exceed max or go below min.

### Networking Essentials

- **Pod-to-Pod Communication:** Every pod gets a unique IP. Pods can communicate with any other pod directly (via the cluster's network fabric, e.g., CNI plugin like Calico, Flannel).
- **Services:** Stable abstraction exposing a set of pods.
  - **ClusterIP (default):** Exposes service on an internal IP in the cluster. Only reachable from within the cluster.
  - **NodePort:** Exposes service on each node's IP at a static port (30000-32767). Accessible from outside the cluster via `<NodeIP>:<NodePort>`.
  - **LoadBalancer:** Creates an external load balancer (cloud provider specific) that routes to the service.
  - **Headless Service (`clusterIP: None`):** Used for StatefulSets to provide stable DNS names for each pod.

**Example: Expose an inference service via LoadBalancer**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  type: LoadBalancer
  selector:
    app: sentiment-model
  ports:
  - protocol: TCP
    port: 80          # port the service listens on
    targetPort: 8080  # port the container listens on
```
**Explanation:** When applied on a cloud provider (EKS, AKS, GKE), this provisions a cloud load balancer with a public IP that forwards traffic to pods with label `app: sentiment-model`.

### Storage for ML

ML workloads need persistent storage for:
- Datasets (training/validation data)
- Model artifacts (checkpoints, final models)
- Logs and metrics

Kubernetes abstracts storage via:
- **PersistentVolume (PV):** A piece of storage in the cluster provisioned by an administrator or dynamically via StorageClass.
- **PersistentVolumeClaim (PVC):** A request for storage by a user. Pods use PVCs as volumes.
- **StorageClass:** Defines different classes of storage (e.g., SSD, HDD, network storage).

**Example: Dynamically provision SSD storage with a PVC**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fast-data
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd  # must exist in cluster
  resources:
    requests:
      storage: 100Gi
```

**Pod using PVC:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  containers:
  - name: trainer
    image: myapp/trainer:latest
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: fast-data
```

**Access Modes:**
- **ReadWriteOnce (RWO):** Can be mounted as read-write by a single node.
- **ReadOnlyMany (ROX):** Mounted as read-only by many nodes.
- **ReadWriteMany (RWX):** Mounted as read-write by many nodes. Useful for shared datasets (requires shared filesystem like NFS, EFS, or GlusterFS).

**Tip for large models:** Use `initContainer` to download the model from blob storage into a volume (like `emptyDir` or PVC) to avoid baking model into image. For even faster startup, consider using a sidecar that caches models or a CSI driver that mounts object storage directly (e.g., AWS EFS CSI driver, Azure File CSI driver).

---

## 2. Distributed Systems & Big Data

### Apache Spark

Spark is a unified analytics engine for large-scale data processing. It's widely used in ML pipelines for feature engineering, data preprocessing, and even distributed training (via MLlib).

#### Key Concepts
- **Driver:** The process that runs the main() function of the application and creates the SparkContext. It schedules tasks and coordinates execution.
- **Executors:** Worker processes that run tasks and store data for cached RDDs/DataFrames.
- **RDD (Resilient Distributed Dataset):** Immutable, partitioned collection of records that can be operated on in parallel. Low-level API.
- **DataFrame:** Distributed collection of rows with a schema. Built on top of RDDs, but with higher-level optimizations via Catalyst optimizer. Preferred for ML/data workflows.
- **Dataset:** Type-safe version of DataFrame (JVM only). Not available in PySpark.
- **DAG (Directed Acyclic Graph):** Spark builds a DAG of stages for each job, then executes tasks.
- **Shuffle:** The process of redistributing data across partitions (e.g., during groupBy, join). Expensive due to disk I/O and network transfer.

#### Spark Application Structure

**Example: PySpark feature engineering pipeline**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# 1. Initialize Spark session
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .config("spark.sql.adaptive.enabled", "true") \  # Dynamic optimization
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .getOrCreate()

# 2. Read data (from Parquet, CSV, etc.)
df = spark.read.parquet("s3://data/raw/transactions.parquet")

# 3. Explore and clean
df.printSchema()
df.describe().show()
df = df.dropna(subset=["user_id", "amount"])

# 4. Feature engineering with DataFrame transformations
df = df.withColumn("is_high_value", when(col("amount") > 1000, 1).otherwise(0))

# 5. User Defined Function (UDF) for custom logic
def category_to_score(category):
    mapping = {"A": 1.0, "B": 2.0, "C": 3.0}
    return mapping.get(category, 0.0)

category_to_score_udf = udf(category_to_score, DoubleType())
df = df.withColumn("category_score", category_to_score_udf(col("category")))

# 6. Assemble features into a vector for ML
feature_cols = ["amount", "category_score", "is_high_value"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vector = assembler.transform(df)

# 7. Scale features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(df_vector)
df_scaled = scaler_model.transform(df_vector)

# 8. Train/test split
train, test = df_scaled.randomSplit([0.8, 0.2], seed=42)

# 9. ML model (Random Forest)
rf = RandomForestClassifier(labelCol="label", featuresCol="scaled_features",
                            numTrees=50, maxDepth=5)
model = rf.fit(train)

# 10. Evaluate
predictions = model.transform(test)
predictions.select("label", "prediction").show(5)

# 11. Save model
model.save("s3://models/rf_model_v1")
```

**Explanation of key points:**
- `spark.sql.adaptive` enables dynamic optimization (e.g., coalescing shuffle partitions, skew join handling).
- UDFs are convenient but can be slow (serialization overhead). For performance, use built-in functions or pandas UDFs (vectorized).
- `VectorAssembler` combines feature columns into a single vector column required by MLlib algorithms.
- StandardScaler normalizes features to zero mean and unit variance (important for many ML models).

#### Spark Internals: Memory Management & Tuning

Spark divides executor memory into:
- **Execution Memory:** Used for shuffles, joins, sorts, aggregations.
- **Storage Memory:** Used for caching user data (RDD/DataFrame persist).
- **Reserved Memory:** Overhead.

If execution and storage compete, Spark uses a unified memory manager; they can borrow from each other.

**Common tuning parameters:**
```python
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.cores", "4")
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.sql.shuffle.partitions", "200")  # default; increase for large data
spark.conf.set("spark.memory.fraction", "0.6")  # fraction of (heap - 300MB) for execution+storage
spark.conf.set("spark.memory.storageFraction", "0.5")  # fraction of above for storage
```

#### Debugging Spark Performance Issues

**1. Data Skew**
- Symptom: Some tasks take much longer than others.
- Detection: Look at Spark UI "Stages" tab; see min/median/max task duration. If max >> median, skew.
- Fixes:
  - Salting: Add random prefix to join key to distribute load.
  - Broadcast join for small tables.
  - Increase shuffle partitions.

**Code to detect skew:**
```python
# Check distribution of key
df.groupBy("join_key").count().orderBy("count", ascending=False).show(10)
```

**Salting example:**
```python
from pyspark.sql.functions import concat, lit, rand, col

# Add salt to skewed key
salted_df = large_df.withColumn(
    "salted_key", 
    concat(col("skewed_key"), lit("_"), (rand() * 10).cast("int"))
)

# Broadcast small table? No, but we can replicate small table with salting too
from pyspark.sql.functions import explode, array

small_df_salted = small_df.withColumn(
    "salt", explode(array([lit(i) for i in range(10)]))
).withColumn(
    "salted_key", concat(col("join_key"), lit("_"), col("salt"))
)

# Now join on salted_key
result = salted_df.join(small_df_salted, "salted_key")
```

**2. Spilling to Disk**
- Symptom: Tasks writing large amounts of data to disk (visible in Spark UI under "Shuffle Spill (Disk)").
- Causes: Not enough memory for operations.
- Fixes: Increase executor memory, reduce partition size (more partitions), use Kryo serialization.

**3. Too Many Small Files**
- Symptom: Reading from many small files causes overhead.
- Fix: Use `coalesce()` or `repartition()` to control output partitions, or enable Adaptive Query Execution (AQE) to coalesce shuffle partitions automatically.

### Apache Flink

Flink is a stream processing framework for stateful computations over unbounded and bounded data streams. Key differentiators from Spark Streaming:
- True streaming (event-at-a-time processing) with low latency.
- Exactly-once semantics for state consistency.
- Event time processing with watermarks.

#### Core Concepts
- **Stream:** An endless flow of events.
- **Operator:** Transformation applied to streams (map, filter, window).
- **Keyed Stream:** Stream partitioned by a key, allowing state per key.
- **State:** Maintained across events (e.g., counts, aggregations). Flink manages state fault-tolerantly via checkpoints.
- **Checkpoint:** Snapshot of operator state and offsets, saved to durable storage. Used for recovery.
- **Watermark:** Mechanism to track event time progress and handle late data.

**Example: PyFlink streaming word count with event time windows**
```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.expressions import col, lit
from pyflink.table.window import Tumble

# 1. Set up streaming environment
env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# 2. Define source table with event time attribute
source_ddl = """
    CREATE TABLE user_clicks (
        user_id STRING,
        page STRING,
        click_time TIMESTAMP(3),
        WATERMARK FOR click_time AS click_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'clicks',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'json',
        'scan.startup.mode' = 'latest-offset'
    )
"""
t_env.execute_sql(source_ddl)

# 3. Define sink table (MySQL)
sink_ddl = """
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
"""
t_env.execute_sql(sink_ddl)

# 4. Execute streaming query: 10-minute tumbling windows
t_env.sql_query("""
    SELECT 
        page,
        TUMBLE_START(click_time, INTERVAL '10' MINUTE) AS window_start,
        TUMBLE_END(click_time, INTERVAL '10' MINUTE) AS window_end,
        COUNT(*) AS view_count
    FROM user_clicks
    GROUP BY page, TUMBLE(click_time, INTERVAL '10' MINUTE)
""").execute_insert("page_views")
```

**Explanation:**
- `WATERMARK` defines how late events can be (5 seconds). Flink will wait 5 seconds after seeing a timestamp before considering the window complete, allowing for out-of-order data.
- Tumbling windows group events into fixed-size, non-overlapping intervals.
- The query is continuous; results are updated as data arrives.

**Checkpointing configuration (for fault tolerance):**
```python
env.enable_checkpointing(60000)  # checkpoint every 60 seconds
env.get_checkpoint_config().set_checkpoint_storage_dir("hdfs:///flink/checkpoints")
env.get_checkpoint_config().set_min_pause_between_checkpoints(30000)
env.get_checkpoint_config().set_checkpoint_timeout(600000)
env.get_checkpoint_config().set_max_concurrent_checkpoints(1)
env.get_checkpoint_config().enable_externalized_checkpoints(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
)
```

### Ray

Ray is a distributed computing framework designed for AI/ML applications. It provides:
- **Tasks:** Remote functions that execute on a cluster.
- **Actors:** Stateful workers that can hold mutable state.
- **Libraries:** Ray Tune (hyperparameter tuning), Ray Serve (model serving), Ray Train (distributed training), RLlib (reinforcement learning).

#### Why Ray for ML Platforms?
Ray simplifies distributed Python. Data scientists can scale their code with minimal changes. It integrates with popular ML libraries (PyTorch, TensorFlow, XGBoost).

#### Ray Tasks (Stateless)

```python
import ray

# Initialize Ray (connect to existing cluster)
ray.init(address="auto")

@ray.remote
def preprocess_chunk(file_path):
    """Simulate preprocessing a chunk of data."""
    import time
    time.sleep(2)  # simulate work
    return f"Processed {file_path}"

# Launch many tasks in parallel
file_paths = [f"data_{i}.csv" for i in range(100)]
futures = [preprocess_chunk.remote(path) for path in file_paths]

# Wait for results
results = ray.get(futures)
print(f"Processed {len(results)} files")
```

**Explanation:**
- `@ray.remote` decorator turns a function into a task that can run on any Ray worker.
- `preprocess_chunk.remote()` returns an `ObjectRef` (future) immediately.
- `ray.get()` blocks until all results are ready and returns them.

#### Ray Actors (Stateful)

```python
@ray.remote
class ModelReplica:
    """Actor that holds a model in memory and serves predictions."""
    
    def __init__(self, model_id, model_path):
        self.model_id = model_id
        self.model = self._load_model(model_path)
        self.request_count = 0
    
    def _load_model(self, path):
        # In reality, load from disk (e.g., torch.load, pickle)
        print(f"Loading model {self.model_id} from {path}")
        return {"name": f"model-{self.model_id}", "loaded": True}
    
    def predict(self, input_data):
        self.request_count += 1
        # Simulate inference
        return {
            "prediction": f"result_{input_data}",
            "model_id": self.model_id,
            "request_count": self.request_count
        }
    
    def get_stats(self):
        return {
            "model_id": self.model_id,
            "request_count": self.request_count
        }

# Create 3 actors (e.g., one per GPU)
actors = [ModelReplica.remote(i, f"/models/model_{i}.bin") for i in range(3)]

# Send 100 inference requests, round-robin
futures = []
for i in range(100):
    actor = actors[i % 3]
    futures.append(actor.predict.remote(f"input_{i}"))

# Collect results
results = ray.get(futures)

# Check stats of first actor
print(ray.get(actors[0].get_stats.remote()))
```

**Explanation:**
- Actors are created with `ModelReplica.remote()`. They hold state (the model, request count) across calls.
- Each call to `predict.remote()` is a task executed by that specific actor.

#### Ray Serve for Model Serving
Ray Serve is a scalable model serving library built on Ray.

```python
from ray import serve
import requests

# Start Ray Serve
serve.start(detached=True)

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 0.5}  # each replica uses 0.5 GPU
)
class SentimentModel:
    def __init__(self):
        # Load model once at startup
        from transformers import pipeline
        self.model = pipeline("sentiment-analysis")
    
    async def __call__(self, request):
        text = await request.body()
        result = self.model(text.decode())[0]
        return {"label": result["label"], "score": result["score"]}

# Deploy the model
SentimentModel.deploy()

# Send a test request
resp = requests.post("http://localhost:8000/SentimentModel", data="I love this!")
print(resp.json())
```

**Explanation:**
- `@serve.deployment` defines a Ray Serve deployment with 2 replicas.
- Each replica has its own instance of the model.
- Requests are load-balanced across replicas.
- Ray Serve handles autoscaling, health checking, and versioning.

### Distributed System Principles

**Key concepts to discuss in interviews:**
- **Consistency:** In distributed storage (e.g., etcd) we often use consensus algorithms like Raft to ensure strong consistency. For ML platforms, consistency of model versions and metadata is critical.
- **Partitioning (Sharding):** How data is split across nodes (e.g., by user ID in Kafka, by hash of key in Spark). Discuss trade-offs: even distribution vs. data locality.
- **Replication:** For fault tolerance (e.g., Spark executors can fail; tasks are re-run). In Flink, state is checkpointed.
- **Fault Tolerance:** At-least-once vs. exactly-once semantics. Spark Streaming offers exactly-once with appropriate sinks; Flink guarantees exactly-once for state.
- **CAP Theorem:** In event of network partition (P), you must choose between availability (A) and consistency (C). Many ML systems choose availability (e.g., model serving) but need consistency for metadata (e.g., model registry).

---

## 3. MLOps & ML Lifecycle

### MLflow

MLflow is an open-source platform for managing the ML lifecycle. It has four components:

#### 1. MLflow Tracking
Logs parameters, metrics, and artifacts from ML runs. Provides a UI to compare experiments.

**Example: Tracking with MLflow**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set tracking URI (where to store runs)
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("house_price_prediction")

with mlflow.start_run(run_name="random_forest_v1"):
    # Log parameters
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    
    # Log model (automatic environment capture)
    mlflow.sklearn.log_model(model, "model")
    
    # Log arbitrary artifacts (e.g., feature importance plot)
    mlflow.log_artifact("feature_importance.png")
    
    # Register model in Model Registry
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", 
                          "HousePriceModel")
```

**Explanation:**
- `mlflow.start_run()` creates a new run. Everything logged inside is associated with that run.
- Parameters, metrics, and artifacts are searchable in the UI.
- `mlflow.sklearn.log_model` saves the model in a format that includes the environment (conda.yaml, requirements.txt) for reproducibility.
- `mlflow.register_model` adds the model to the registry with a given name.

#### 2. MLflow Projects
Packages ML code in a reusable, reproducible format. A project is defined by an `MLproject` file.

**Example MLproject file:**
```yaml
name: house_price_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 5}
    command: "python train.py --n_estimators {n_estimators} --max_depth {max_depth}"
```

**Run a project:**
```bash
mlflow run . -P n_estimators=200 -P max_depth=10
```

#### 3. MLflow Models
A standard format for packaging ML models that can be used by various serving tools. Models can be in "flavors" (python_function, sklearn, pytorch, tensorflow).

**Load a model for inference:**
```python
import mlflow.pyfunc

# Load as generic Python function
model = mlflow.pyfunc.load_model("models:/HousePriceModel/Production")
predictions = model.predict(data)
```

#### 4. MLflow Model Registry
A centralized model store with versioning, stage transitions (Staging, Production, Archived), and annotations.

**Transition a model to Production via API:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="HousePriceModel",
    version=3,
    stage="Production"
)
```

**Model versioning in interview context:**
- Each registered model has versions.
- A version can be in one of three stages: None, Staging, Production, Archived.
- You can load a model by alias: `models:/HousePriceModel/Production` always gets the current production version.
- This enables safe rollbacks (just change the stage assignment).

### CI/CD for ML

Traditional CI/CD pipelines test and deploy code. ML pipelines must also test data and models.

**Key differences:**
- **Data validation:** Ensure new data has the same schema and distribution as training data.
- **Model validation:** Evaluate model performance against a baseline before deployment.
- **Model deployment:** Canary deployments, A/B testing.
- **Monitoring:** Detect data drift, concept drift, performance degradation.

**Example GitLab CI pipeline for ML:**
```yaml
stages:
  - data-validation
  - training
  - model-validation
  - deployment

variables:
  MLFLOW_TRACKING_URI: "http://mlflow:5000"

# Stage 1: Validate data schema and quality
data-validation:
  stage: data-validation
  script:
    - python scripts/validate_data.py --data-path data/raw/ --schema schema.json
    - python scripts/check_drift.py --reference data/train/stats.json
  artifacts:
    reports:
      html: validation_report.html

# Stage 2: Train model (only on main branch)
training:
  stage: training
  script:
    - python scripts/train.py --config configs/model_config.yaml
  artifacts:
    paths:
      - models/
      - mlruns/
  only:
    - main

# Stage 3: Validate model performance against baseline
model-validation:
  stage: model-validation
  script:
    - python scripts/evaluate.py --model-path models/latest/ --test-data data/test/
    - python scripts/compare_with_baseline.py --new-metrics metrics.json --threshold 0.01
  dependencies:
    - training
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'

# Stage 4: Deploy to staging (automated) and production (manual)
deploy-staging:
  stage: deployment
  script:
    - python scripts/package_model.py --model-path models/latest/ --output packaged_model/
    - kubectl apply -f k8s/staging/deployment.yaml
    - python scripts/smoke_test.py --endpoint staging.example.com
  environment:
    name: staging
  only:
    - main

deploy-production:
  stage: deployment
  script:
    - python scripts/promote_model.py --model-name HousePriceModel --stage Production
    - kubectl apply -f k8s/production/deployment.yaml
    - python scripts/smoke_test.py --endpoint production.example.com
  environment:
    name: production
  when: manual  # requires human approval
  only:
    - main
```

**Explanation of key steps:**
- `data-validation` ensures data quality before training. If data is bad, the pipeline fails early.
- `training` runs only on main branch to avoid wasting resources on feature branches.
- `model-validation` compares new model metrics with the current production model. If improvement is below threshold, pipeline can fail.
- `deploy-staging` deploys to a staging environment for further testing.
- `deploy-production` is manual, allowing a human to approve after staging validation.

### Model Serving Patterns

**Batch Inference:**
- Run predictions on a large set of data at scheduled intervals.
- Use Spark, Flink, or Ray to distribute the workload.
- Output to database or data lake.

**Online Inference (Real-time):**
- Single prediction requests with low latency.
- Deploy models as microservices (REST/gRPC) using frameworks like TorchServe, TensorFlow Serving, NVIDIA Triton, or Ray Serve.
- Often requires GPUs, so autoscaling based on GPU utilization is key.

**Hybrid Approaches:**
- **Pre-warming:** Keep models loaded in memory across replicas to avoid cold starts.
- **Model caching:** Use a sidecar cache (e.g., Redis) to store frequent inference results.

---

## 4. Reliability Engineering & Performance

### SRE Principles Applied to ML Platforms

- **SLI (Service Level Indicator):** A quantitative measure of some aspect of the service (e.g., inference latency p99, error rate, throughput).
- **SLO (Service Level Objective):** Target value or range for an SLI (e.g., p99 latency < 100ms for 99% of requests over a rolling 30-day window).
- **Error Budget:** 1 - SLO (e.g., if SLO is 99.9% availability, error budget is 0.1%). You can spend this budget on deployments, experiments, etc. Once exhausted, you slow down changes to focus on stability.

**For ML platforms:**
- Model accuracy can also be an SLI (e.g., accuracy drift). If accuracy drops below a threshold, that's an error budget burn.
- Data freshness (how recent is the data used for inference) is another SLI.

### Performance Analysis: USE and RED Methods

**USE Method (for infrastructure):**
- **Utilization:** Average time resource was busy (e.g., CPU 70%).
- **Saturation:** Degree of extra work queued (e.g., load average, run queue length).
- **Errors:** Count of error events.

**RED Method (for services):**
- **Rate:** Requests per second.
- **Errors:** Number of failed requests.
- **Duration:** Distribution of latency (p50, p95, p99).

**Example troubleshooting flow using RED:**
1. Alert: p99 latency increased from 50ms to 500ms.
2. Check Rate: Is traffic spiking? (Could be overload)
3. Check Errors: Are there many 5xx errors? (Could be service crashing)
4. Drill down: Which endpoint is slow? Which pod? Use distributed tracing.
5. Check infrastructure (USE): Is CPU saturated? Is network dropping packets?
6. If CPU high, maybe need more replicas. If memory high, maybe GC issues.
7. If no resource issue, check model complexity (maybe new model version is slower).

### Troubleshooting Frameworks

**General approach (adapt to any problem):**
1. **Identify scope:** Is it one user, one pod, one node, or global?
2. **Check recent changes:** Deployments, config changes, new model versions.
3. **Look at metrics:** Grafana dashboards for RED/USE.
4. **Logs:** Use centralized logging (ELK, Loki) to search for errors.
5. **If stuck, reproduce:** Canary deployment, load test in staging.

**Common ML platform issues and solutions:**

**Issue: Training job runs out of memory**
- Check Spark UI: Look at task memory usage.
- Increase executor memory, or reduce partition size (increase partitions).
- Use Kryo serialization to reduce memory footprint.
- Check for data skew.

**Issue: Inference latency spikes after new model version**
- Compare model complexity (e.g., layer size, precision).
- Check if GPU memory is sufficient; maybe model doesn't fit and spills to CPU.
- Profile the model: use `torch.profiler` or TensorFlow profiling.
- Consider model quantization or pruning.

**Issue: Model version mismatch between training and serving**
- Ensure model artifacts are versioned and stored with MLflow.
- Use the same model URI in both training and serving.
- In Kubernetes, use initContainer to download the exact model version from registry.

---

## 5. Interview Question Templates & Scenarios

### STAR Method for Behavioral Questions
- **Situation:** Set the context.
- **Task:** What needed to be done?
- **Action:** What specific steps did you take?
- **Result:** What was the outcome? (Quantify if possible)

### Common Scenario Questions

**Q: "A user reports that their Spark training job is running 10x slower than usual. Walk me through your debugging process."**

**A (structured):**
- **Situation:** A data scientist complains their job, which usually takes 1 hour, now takes 10 hours.
- **Task:** Identify root cause and restore performance.
- **Action:**
  1. **Check job parameters:** Any code changes? Look at Git history. If no code changes, suspect data or cluster issues.
  2. **Check cluster health:** `kubectl top nodes` – any nodes under pressure? `kubectl describe nodes` for conditions.
  3. **Check Spark UI:**
     - Look at "Stages" tab: Are there long-running tasks? Check if data skew (max >> median duration).
     - "Storage" tab: Is data cached? If not, maybe caching was removed.
     - "SQL" tab: See the physical plan – any broadcast joins? Skew joins?
  4. **Check executor logs:** Look for excessive GC logs (`Full GC`), which indicate memory pressure.
  5. **If data skew:** Compute distribution of key; if skewed, suggest salting or broadcast join.
  6. **If memory pressure:** Check if `spark.sql.adaptive.enabled` is on; it can help.
  7. **If no improvement:** Compare with a previous successful run's configuration. Maybe cluster was downgraded or resource quotas changed.
- **Result:** Found that a new join key was heavily skewed; implemented salting, reducing runtime back to 1 hour.

**Q: "How would you design a multi-tenant ML platform on Kubernetes?"**

**A:**
- **Namespaces per team:** Isolate resources logically.
- **ResourceQuotas:** Cap each team's CPU/memory usage.
- **LimitRanges:** Prevent individual pods from hogging resources.
- **PriorityClasses:** Production jobs can preempt experimental ones.
- **Network Policies:** Restrict traffic between namespaces.
- **Node Pools:** Separate GPU nodes for inference (with taints) from CPU nodes for batch jobs. Use node affinity to schedule inference pods on GPU nodes.
- **Monitoring:** Prometheus metrics per namespace, with dashboards showing usage vs quota.
- **Model Registry:** Central MLflow instance accessible to all teams, with access control via RBAC.

**Q: "Explain how you would ensure high availability for a model serving endpoint."**

**A:**
- **Multiple replicas:** Deploy with at least 2 replicas spread across nodes (using pod anti-affinity) to avoid single node failure.
- **Load balancer:** Use a Kubernetes Service of type LoadBalancer (or an ingress) to distribute traffic.
- **Health checks:** Configure liveness and readiness probes so that unhealthy pods are removed.
- **Autoscaling:** Use HPA (or KEDA) to scale based on CPU/GPU utilization or request latency.
- **Canary deployments:** Deploy new model versions gradually, monitoring error rate and latency.
- **Failover:** If a whole region fails, have a multi-cluster setup with global load balancer.
- **Model versioning:** Always have the previous model version ready to roll back if the new one fails.

**Q: "What's the difference between Spark and Flink? When would you use each?"**

**A:**
- **Spark** is a batch-first engine that also does micro-batch streaming. It's great for large-scale ETL, data preprocessing, and iterative algorithms (like ML). Use Spark when you need to process large historical datasets and can tolerate seconds of latency.
- **Flink** is a true stream processor with low latency (milliseconds) and exactly-once semantics. Use Flink for real-time analytics, fraud detection, or any application that requires immediate reaction to events.
- **Hybrid approach:** You might use Flink for real-time feature computation and Spark for batch training on accumulated features.

---

## 6. Quick Reference Cheat Sheet

### Kubernetes CLI Shortcuts
```bash
# Get all pods with node info
kubectl get pods -o wide

# Describe a problematic pod
kubectl describe pod <pod-name>

# Tail logs
kubectl logs -f <pod-name>

# Exec into container
kubectl exec -it <pod-name> -- /bin/bash

# Get events sorted by time
kubectl get events --sort-by='.lastTimestamp'
```

### Spark Tuning Cheat Sheet
```python
# Common configs
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "200")  # adjust
spark.conf.set("spark.memory.fraction", "0.6")
spark.conf.set("spark.memory.storageFraction", "0.5")
```

### MLflow CLI
```bash
# Start UI
mlflow ui --port 5000

# Register model
mlflow models register -m runs:/<run-id>/model -n ModelName

# Serve a model locally
mlflow models serve -m models:/ModelName/Production -p 5001
```

### Interview Buzzwords to Use
- "Control plane vs. data plane"
- "Eventually consistent vs. strongly consistent"
- "Data locality"
- "Backpressure" (Flink)
- "Checkpointing"
- "Model staleness"
- "Shadow mode" (deploy new model alongside old, compare outputs without serving)
- "Canary deployment"
- "Multi-tenancy"
- "Resource isolation"
