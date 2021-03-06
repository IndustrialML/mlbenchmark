=============================
ML Pipeline benchmarking
=============================

Benchmarking suite for ML pipelines


Features
--------

AccuracyBenchmark
-----------------

+---------------------------+--------------+-------------+
|                           | **accuracy** | **jaccard** |
+---------------------------+--------------+-------------+
| **env**                   |              |             |
+---------------------------+--------------+-------------+
| **local-python-baseline** | 0.1111       | 0.1111      |
+---------------------------+--------------+-------------+
| **local-python-svm**      | 0.9898       | 0.9898      |
+---------------------------+--------------+-------------+
| **local-python-forest**   | 0.9494       | 0.9494      |
+---------------------------+--------------+-------------+

SequentialLoadBenchmark
-----------------

+---------------------------+----------+------------+---------+---------+---------+-------------+
|                           | **mean** | **median** | **min** | **max** | **std** | **geomean** |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **env**                   |          |            |         |         |         |             |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-baseline** | 0.0024   | 0.0023     | 0.0018  | 0.0042  | 0.0004  | 0.0023      |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-svm**      | 0.0025   | 0.0025     | 0.0019  | 0.0042  | 0.0004  | 0.0025      |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-forest**   | 0.0026   | 0.0026     | 0.0019  | 0.0039  | 0.0003  | 0.0026      |
+---------------------------+----------+------------+---------+---------+---------+-------------+

ConcurrentLoadBenchmark
-----------------------

+---------------------------+----------+------------+---------+---------+---------+-------------+
|                           | **mean** | **median** | **min** | **max** | **std** | **geomean** |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **env**                   |          |            |         |         |         |             |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-baseline** | 0.0029   | 0.0028     | 0.0020  | 0.0092  | 0.0007  | 0.0029      |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-svm**      | 0.0028   | 0.0026     | 0.0021  | 0.0105  | 0.0008  | 0.0026      |
+---------------------------+----------+------------+---------+---------+---------+-------------+
| **local-python-forest**   | 0.0026   | 0.0024     | 0.0019  | 0.0092  | 0.0007  | 0.0025      |
+---------------------------+----------+------------+---------+---------+---------+-------------+