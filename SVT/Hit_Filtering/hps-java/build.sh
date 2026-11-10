#!/bin/bash
# Compile the SVT hit-filter driver and its parity test against an hps-java distribution jar.
#   ./build.sh [path/to/hps-distribution-*-bin.jar]
# Classes go to build/classes; a jar with the driver to build/hps-svt-hitfilter.jar (add it to the class path
# of hps-java, e.g. java -cp hps-distribution-bin.jar:build/hps-svt-hitfilter.jar org.hps.job.JobManager ...).
set -e
cd "$(dirname "$0")"
JAR=${1:-/net/home/erbaz/hps-java/distribution/target/hps-distribution-5.3.1-SNAPSHOT-bin.jar}
mkdir -p build/classes
javac -Xlint:unchecked -cp "$JAR" -d build/classes $(find src -name "*.java")
jar cf build/hps-svt-hitfilter.jar -C build/classes org
echo "built build/hps-svt-hitfilter.jar against $JAR"
