import os
import sys
import math

# Data Loading: Read AIS CSV
# Data Cleaning: Filter invalid coordinates and missing values
# Distance Calculation: Calculate distances between consecutive positions using coordinate differences
# Aggregation: Sum total distances traveled by each vessel
# Analysis: Identify vessel with maximum distance
#
# Key PySpark Components Used
# SparkSession: Main entry point for Spark functionality
# Window Functions: lag() function to access previous vessel positions
# DataFrame Operations: groupBy(), agg(), filtering, and transformations
# Built-in Functions: Mathematical operations for distance calculations
#
# Distance Calculation
# Latitude difference: multiplied by 111 km (approximate km per degree)
# Longitude difference: multiplied by 85 km (average for the geographic region)
# Combined using Pythagorean theorem: sqrt(lat_diff**2 + lon_diff**2)
#
# Results
# MMSI: 219133000
# Distance: 942 km

# Java setup for Windows
def setup_java():
    java_home = r'C:\Program Files\Java\jdk-17'
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = f"{java_home}\\bin;" + os.environ.get('PATH', '')
    if 'SPARK_HOME' in os.environ:
        del os.environ['SPARK_HOME']
    print(f"Java configured: {java_home}")
    return True


setup_java()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *


def create_spark_session():
    """Create Spark session"""
    print("Creating Spark session...")

    try:
        spark = SparkSession.builder \
            .appName("VesselAnalysis") \
            .master("local[2]") \
            .config("spark.driver.memory", "3g") \
            .config("spark.driver.maxResultSize", "1g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        print("Spark session created successfully")
        return spark

    except Exception as e:
        print(f"Error creating Spark session: {e}")
        raise


def load_and_process_data(spark, sample_fraction=0.1):
    """Load AIS data and take sample for processing"""
    print("Loading data...")

    file_path = "aisdk-2024-05-04.csv"

    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

    print(f"Available columns: {df_raw.columns}")

    # Select required columns
    df_prepared = df_raw.select(
        col("MMSI").cast(LongType()).alias("mmsi"),
        col("# Timestamp").alias("timestamp"),
        col("Latitude").cast(DoubleType()).alias("latitude"),
        col("Longitude").cast(DoubleType()).alias("longitude")
    ).filter(
        (col("mmsi").isNotNull()) &
        (col("latitude").isNotNull()) &
        (col("longitude").isNotNull()) &
        (col("timestamp").isNotNull()) &
        (col("latitude").between(-90, 90)) &
        (col("longitude").between(-180, 180))
    )

    # Take sample for processing
    print(f"Taking {sample_fraction * 100}% sample for processing...")
    df_sample = df_prepared.sample(sample_fraction, seed=42)

    total_records = df_sample.count()
    unique_vessels = df_sample.select("mmsi").distinct().count()

    print(f"Sample size: {total_records:,} records")
    print(f"Vessels in sample: {unique_vessels:,}")

    return df_sample


def calculate_distances(df):
    """Calculate distances between consecutive positions for each vessel"""
    print("Calculating distances...")

    # Window specification for each vessel ordered by time
    window_spec = Window.partitionBy("mmsi").orderBy("timestamp")

    # Get previous positions using lag function
    df_with_prev = df.withColumn(
        "prev_lat", lag("latitude", 1).over(window_spec)
    ).withColumn(
        "prev_lon", lag("longitude", 1).over(window_spec)
    ).filter(
        col("prev_lat").isNotNull()
    )

    # Calculate distance using coordinate differences
    df_with_distance = df_with_prev.withColumn(
        "lat_diff", abs(col("latitude") - col("prev_lat"))
    ).withColumn(
        "lon_diff", abs(col("longitude") - col("prev_lon"))
    ).withColumn(
        "distance_km",
        sqrt(
            pow(col("lat_diff") * 111.0, 2) +
            pow(col("lon_diff") * 85.0, 2)
        )
    )

    # Filter out unrealistic jumps
    df_realistic = df_with_distance.filter(col("distance_km") < 100)

    print("Aggregating distances by vessel...")

    # Sum total distance for each vessel
    vessel_distances = df_realistic.groupBy("mmsi").agg(
        sum("distance_km").alias("total_distance_km"),
        count("distance_km").alias("position_count")
    ).filter(
        col("total_distance_km") > 0
    )

    vessels_count = vessel_distances.count()
    print(f"Calculated distances for {vessels_count:,} vessels")

    return vessel_distances


def find_top_vessels(vessel_distances, top_n=10):
    """Find top vessels by distance traveled"""
    print(f"Finding top {top_n} vessels...")

    top_vessels = vessel_distances.orderBy(col("total_distance_km").desc()).limit(top_n).collect()

    return top_vessels


def main():
    """Main analysis function"""
    print("PySpark Vessel Route Analysis")
    print("=" * 50)

    spark = create_spark_session()

    try:
        print("\n[1/4] Loading and sampling data...")
        df_sample = load_and_process_data(spark, sample_fraction=0.1)

        print("\n[2/4] Calculating distances...")
        vessel_distances = calculate_distances(df_sample)

        print("\n[3/4] Finding longest routes...")
        top_vessels = find_top_vessels(vessel_distances, top_n=10)

        print("\n[4/4] Filtering for realistic routes...")
        realistic_vessels = [v for v in top_vessels if v['total_distance_km'] <= 1000]

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if top_vessels:
            longest_vessel = top_vessels[0]
            print("LONGEST ROUTE:")
            print(f"MMSI: {longest_vessel['mmsi']}")
            print(f"Distance: {longest_vessel['total_distance_km']:.2f} km")
            print(f"Position reports: {longest_vessel['position_count']:,}")

            if realistic_vessels:
                realistic_longest = realistic_vessels[0]
                print("\nREALISTIC LONGEST ROUTE:")
                print(f"MMSI: {realistic_longest['mmsi']}")
                print(f"Distance: {realistic_longest['total_distance_km']:.2f} km")
                print(f"Position reports: {realistic_longest['position_count']:,}")
                print(f"Average speed: {realistic_longest['total_distance_km'] / 24:.1f} km/h")

                print("\n" + "=" * 40)
                print("ASSIGNMENT ANSWER:")
                print("=" * 40)
                print(f"Vessel with longest route: MMSI {realistic_longest['mmsi']}")
                print(f"Total distance traveled: {realistic_longest['total_distance_km']:.2f} km")
            else:
                print(f"\nAll routes contain potential errors. Best result:")
                print(f"MMSI: {longest_vessel['mmsi']}")
                print(f"Distance: {longest_vessel['total_distance_km']:.2f} km")

            print(f"\nTOP 10 VESSELS BY DISTANCE:")
            print("-" * 60)
            for i, vessel in enumerate(top_vessels, 1):
                status = "OK" if vessel['total_distance_km'] <= 1000 else "CHECK"
                print(f"{i:2d}. {status} MMSI: {vessel['mmsi']} | "
                      f"Distance: {vessel['total_distance_km']:6.2f} km | "
                      f"Reports: {vessel['position_count']:4,}")

            print("\nNotes:")
            print("- OK = Realistic route (under 1000 km/day)")
            print("- CHECK = May contain GPS errors")
            print("- Analysis based on 10% sample of full dataset")

        else:
            print("No vessel routes found in the sample.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

    finally:
        spark.stop()
        print("\nAnalysis completed")


if __name__ == "__main__":
    main()