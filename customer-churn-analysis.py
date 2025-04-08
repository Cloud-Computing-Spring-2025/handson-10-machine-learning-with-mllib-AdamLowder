from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):

    # Fill missing TotalCharges with 0
    df = df.fillna({'TotalCharges': 0})

    # Define categorical columns to be indexed and one-hot encoded
    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_Vec") for col in categorical_cols]

    # Define numeric columns
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    # Combine all feature columns
    feature_cols = [col + "_Vec" for col in categorical_cols] + numeric_cols

    # Assemble features into a single feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Convert Churn column to label index
    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")

    # Build and apply the preprocessing pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, label_indexer])
    model = pipeline.fit(df)
    final_df = model.transform(df).select("features", "label")

    return final_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):

    # Split the data into training and test sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

    # Initialize and train the logistic regression model
    lr = LogisticRegression(featuresCol='features', labelCol='label')
    lr_model = lr.fit(train_data)

    # Make predictions on test data
    predictions = lr_model.transform(test_data)

    # Evaluate model using AUC (Area Under the ROC Curve)
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {auc:.4f}")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):

    # Apply Chi-Square feature selector to pick top 5 features
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)

    # Display selected features and label
    result.select("selectedFeatures", "label").show(5, truncate=False)
   
# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):

    # Split data again for model tuning
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator()

    # Define models to compare
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBT": GBTClassifier()
    }

    # Define hyperparameter grids for each model
    param_grids = {
        "Logistic Regression": ParamGridBuilder()
            .addGrid(LogisticRegression.regParam, [0.01, 0.1])
            .build(),

        "Decision Tree": ParamGridBuilder()
            .addGrid(DecisionTreeClassifier.maxDepth, [3, 5, 10])
            .build(),

        "Random Forest": ParamGridBuilder()
            .addGrid(RandomForestClassifier.numTrees, [10, 20])
            .build(),

        "GBT": ParamGridBuilder()
            .addGrid(GBTClassifier.maxIter, [10, 20])
            .build()
    }

    # Train, evaluate, and compare each model
    for name, model in models.items():
        print(f"\nTuning {name}...")
        param_grid = param_grids[name]

        cv = CrossValidator(
            estimator=model,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5
        )

        cv_model = cv.fit(train_data)
        predictions = cv_model.transform(test_data)
        auc = evaluator.evaluate(predictions)

        best_params = {k.name: v for k, v in cv_model.bestModel.extractParamMap().items()}
        print(f"Best AUC for {name}: {auc:.4f}")
        print(f"Best Hyperparameters: {best_params}")
    

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
