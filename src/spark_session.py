"""Shared SparkSession builder for the bigdata-trading project."""
from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(
    app_name: str,
    driver_mem: str = "4g",
    shuffle_parts: int = 16,
) -> SparkSession:
    """Return a configured local SparkSession.

    Parameters
    ----------
    app_name : str
        Application name shown in the Spark UI.
    driver_mem : str
        Driver memory (e.g. "4g"). Local mode uses driver as executor.
    shuffle_parts : int
        spark.sql.shuffle.partitions. Keep low for local laptop runs;
        the default of 200 is wasteful for ~2M-row datasets.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
