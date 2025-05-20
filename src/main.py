from trainer import SparkConfig, Trainer
from random_forest import RandomForestModel
from transforms import Transforms, Normalize

# Computed mean/std for Cancer Data
transforms = Transforms([
    Normalize(
        mean=[0]*30,
        std=[1]*30
    )
])

if __name__ == "__main__":
    spark_config = SparkConfig()
    spark_config.receivers = 4
    spark_config.batch_interval = 10

    rf = RandomForestModel(numTrees=100, maxDepth=5)
    trainer = Trainer(rf, "train", spark_config, transforms)
    trainer.train()
    trainer.predict()