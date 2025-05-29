import os
import sys

# Force Java path for PySpark
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['PATH'] = r'C:\Program Files\Java\jdk-17\bin;' + os.environ.get('PATH', '')

# Verify Java is found
print(f"JAVA_HOME set to: {os.environ['JAVA_HOME']}")
if os.path.exists(os.path.join(os.environ['JAVA_HOME'], 'bin', 'java.exe')):
    print("Java executable found!")
else:
    print("ERROR: Java executable not found!")

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import math


def create_spark_session():
    """
    Initialize Spark session with optimized configuration for maritime data analysis.
    """
    spark = SparkSession.builder \
        .appName("AIS_Vessel_Route_Analysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("Spark session created")
    print(f"Spark version: {spark.version}")

    return spark


def load_ais_data(spark, file_path):
    """
    Load AIS data from CSV file with schema validation.
    """
    print("Loading AIS data...")

    # Define schema for consistent data types
    ais_schema = StructType([
        StructField("MMSI", LongType(), False),
        StructField("BaseDateTime", StringType(), False),
        StructField("LAT", DoubleType(), False),
        StructField("LON", DoubleType(), False),
        StructField("SOG", DoubleType(), True),
        StructField("COG", DoubleType(), True),
        StructField("Heading", DoubleType(), True),
        StructField("VesselName", StringType(), True),
        StructField("IMO", StringType(), True),
        StructField("CallSign", StringType(), True),
        StructField("VesselType", IntegerType(), True),
        StructField("Status", IntegerType(), True),
        StructField("Length", DoubleType(), True),
        StructField("Width", DoubleType(), True),
        StructField("Draft", DoubleType(), True),
        StructField("Cargo", IntegerType(), True)
    ])

    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .schema(ais_schema) \
        .csv(file_path)

    initial_count = df.count()
    print(f"Initial records: {initial_count:,}")

    # Filter invalid coordinates and null values
    df_clean = df.filter(
        (col("LAT").between(-90, 90)) &
        (col("LON").between(-180, 180)) &
        (col("MMSI").isNotNull()) &
        (col("BaseDateTime").isNotNull())
    )

    clean_count = df_clean.count()
    print(f"Valid records: {clean_count:,}")
    print(f"Filtered out: {initial_count - clean_count:,}")

    return df_clean


def prepare_temporal_data(df):
    """
    Convert timestamp strings to datetime and add temporal features.
    """
    print("Processing timestamps...")

    df_temporal = df.withColumn(
        "timestamp",
        to_timestamp(col("BaseDateTime"), "yyyy-MM-dd'T'HH:mm:ss")
    ).filter(col("timestamp").isNotNull())

    df_temporal = df_temporal.withColumn("hour", hour(col("timestamp"))) \
        .withColumn("minute", minute(col("timestamp")))

    # Get data range statistics
    time_stats = df_temporal.agg(
        min("timestamp").alias("earliest"),
        max("timestamp").alias("latest"),
        countDistinct("MMSI").alias("unique_vessels")
    ).collect()[0]

    print(f"Time range: {time_stats['earliest']} to {time_stats['latest']}")
    print(f"Unique vessels: {time_stats['unique_vessels']:,}")

    return df_temporal


def haversine_distance_udf():
    """
    Create UDF for calculating Haversine distance between two geographic points.
    """

    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance in nautical miles.
        """
        if any(x is None for x in [lat1, lon1, lat2, lon2]):
            return 0.0

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in nautical miles
        r_nm = 3440.065

        return c * r_nm

    return udf(haversine, DoubleType())


def calculate_vessel_routes(df):
    """
    Calculate distances between consecutive positions for each vessel.
    """
    print("Calculating vessel routes...")

    # Window for ordering positions by vessel and time
    vessel_window = Window.partitionBy("MMSI").orderBy("timestamp")

    # Add previous position data
    df_with_prev = df.select(
        "MMSI", "timestamp", "LAT", "LON", "VesselName"
    ).withColumn("prev_LAT", lag("LAT").over(vessel_window)) \
        .withColumn("prev_LON", lag("LON").over(vessel_window)) \
        .withColumn("prev_timestamp", lag("timestamp").over(vessel_window))

    haversine_udf = haversine_distance_udf()

    # Calculate segment distances
    df_distances = df_with_prev.withColumn(
        "segment_distance",
        when(
            (col("prev_LAT").isNotNull()) & (col("prev_LON").isNotNull()),
            haversine_udf(col("prev_LAT"), col("prev_LON"), col("LAT"), col("LON"))
        ).otherwise(0.0)
    ).withColumn(
        "time_diff_minutes",
        when(
            col("prev_timestamp").isNotNull(),
            (unix_timestamp("timestamp") - unix_timestamp("prev_timestamp")) / 60
        ).otherwise(0.0)
    )

    # Filter unrealistic speeds (over 50 knots)
    df_realistic = df_distances.filter(
        (col("time_diff_minutes") == 0) |
        (col("segment_distance") / (col("time_diff_minutes") / 60) <= 50)
    )

    print("Applied speed filtering (max 50 knots)")

    return df_realistic


def aggregate_vessel_distances(df_distances):
    """
    Aggregate total distances and statistics by vessel.
    """
    print("Aggregating vessel statistics...")

    vessel_stats = df_distances.groupBy("MMSI") \
        .agg(
        sum("segment_distance").alias("total_distance_nm"),
        count("segment_distance").alias("position_reports"),
        min("timestamp").alias("first_report"),
        max("timestamp").alias("last_report"),
        first("VesselName").alias("vessel_name"),
        avg("segment_distance").alias("avg_segment_distance"),
        max("segment_distance").alias("max_segment_distance")
    ).withColumn(
        "journey_duration_hours",
        (unix_timestamp("last_report") - unix_timestamp("first_report")) / 3600
    ).filter(
        col("total_distance_nm") > 0
    )

    total_vessels = vessel_stats.count()
    print(f"Vessels with movement: {total_vessels:,}")

    return vessel_stats


def find_longest_route(vessel_stats):
    """
    Find the vessel that traveled the longest distance.
    """
    print("Finding longest route...")

    longest_route = vessel_stats.orderBy(col("total_distance_nm").desc()).first()

    print("\n" + "=" * 60)
    print("LONGEST ROUTE RESULTS")
    print("=" * 60)
    print(f"MMSI: {longest_route['MMSI']}")
    print(f"Vessel Name: {longest_route['vessel_name'] or 'Unknown'}")
    print(f"Total Distance: {longest_route['total_distance_nm']:.2f} nautical miles")
    print(f"Journey Duration: {longest_route['journey_duration_hours']:.2f} hours")
    print(f"Position Reports: {longest_route['position_reports']:,}")
    print(f"Average Segment: {longest_route['avg_segment_distance']:.3f} nm")
    print(f"Longest Segment: {longest_route['max_segment_distance']:.3f} nm")
    print(f"First Report: {longest_route['first_report']}")
    print(f"Last Report: {longest_route['last_report']}")

    return longest_route


def generate_summary_statistics(vessel_stats):
    """
    Generate fleet analysis summary statistics.
    """
    print("\n" + "=" * 60)
    print("FLEET ANALYSIS SUMMARY")
    print("=" * 60)

    fleet_summary = vessel_stats.agg(
        count("MMSI").alias("total_vessels"),
        sum("total_distance_nm").alias("fleet_total_distance"),
        avg("total_distance_nm").alias("avg_vessel_distance"),
        stddev("total_distance_nm").alias("distance_std_dev"),
        min("total_distance_nm").alias("min_distance"),
        max("total_distance_nm").alias("max_distance"),
        avg("journey_duration_hours").alias("avg_journey_duration")
    ).collect()[0]

    print(f"Total Active Vessels: {fleet_summary['total_vessels']:,}")
    print(f"Fleet Total Distance: {fleet_summary['fleet_total_distance']:.2f} nm")
    print(f"Average Distance per Vessel: {fleet_summary['avg_vessel_distance']:.2f} nm")
    print(f"Distance Standard Deviation: {fleet_summary['distance_std_dev']:.2f} nm")
    print(f"Shortest Route: {fleet_summary['min_distance']:.2f} nm")
    print(f"Longest Route: {fleet_summary['max_distance']:.2f} nm")
    print(f"Average Journey Duration: {fleet_summary['avg_journey_duration']:.2f} hours")

    # Display top 10 vessels
    print(f"\nTOP 10 VESSELS BY DISTANCE:")
    print("-" * 80)
    top_vessels = vessel_stats.orderBy(col("total_distance_nm").desc()).limit(10)

    for i, vessel in enumerate(top_vessels.collect(), 1):
        vessel_name = vessel['vessel_name'] or 'Unknown'
        print(f"{i:2d}. MMSI: {vessel['MMSI']} | "
              f"Name: {vessel_name[:20]:20s} | "
              f"Distance: {vessel['total_distance_nm']:8.2f} nm | "
              f"Duration: {vessel['journey_duration_hours']:6.2f}h")


def main():
    """
    Main execution function for AIS vessel route analysis.
    """
    print("Starting AIS Vessel Route Analysis")
    print("=" * 60)

    spark = create_spark_session()

    try:
        # Update this path to your actual data file location
        data_file_path = "ais_data_2024_05_04.csv"

        # Data processing pipeline
        ais_data = load_ais_data(spark, data_file_path)
        temporal_data = prepare_temporal_data(ais_data)
        route_data = calculate_vessel_routes(temporal_data)
        vessel_statistics = aggregate_vessel_distances(route_data)

        # Analysis results
        longest_route_vessel = find_longest_route(vessel_statistics)
        generate_summary_statistics(vessel_statistics)

        # Save results
        print(f"\nSaving results...")
        vessel_statistics.coalesce(1).write.mode("overwrite").csv("vessel_analysis_results", header=True)
        print("Results saved to 'vessel_analysis_results' directory")

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        raise

    finally:
        spark.stop()
        print("\nSpark session terminated")


if __name__ == "__main__":
    """
    Entry point for AIS vessel route analysis.

    Requirements:
    - Apache Spark installed
    - PySpark library
    - AIS CSV data file

    Output:
    - MMSI of vessel with longest route
    - Total distance traveled
    - Fleet analysis statistics
    - CSV results file
    """
    main()