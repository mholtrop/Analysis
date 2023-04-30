#!/usr/bin/env bash
#
if [ -z "${1}" ]; then
  echo "Usage: ${0} Name <Nseq> <random init stdev> <activation> <rate> <momentum>"
  exit 1
else
  Name="${1}"
fi
if [ -z "${2}" ]; then
  Nseq=100
else
  Nseq=${2}
fi
if [ -z "${3}" ]; then
  random=""
else
  random="--random "${3}
fi
if [ -z "${4}" ]; then
  activation=""
else
  activation="--activation "${4}
fi
if [ -z "${5}" ]; then
  rate=""
else
  rate="--rate "${5}
fi
if [ -z "${6}" ]; then
  momentum=""
else
  momentum="--momentum "${6}
fi

../MLTrainer_exy.py -m 6 --rate 1.e-5 --file Layers_2_${Name}.json  electrons_n1.feather -d -n ${Nseq}  ${random} ${activation} --freeze [2,3,4,5,6,7,8,9,10] --root
cp Layers_2_${Name}.json Layers_3_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_3_${Name}.json  electrons_n1.feather -d -n ${Nseq}  ${random} ${activation} --freeze [3,4,5,6,7,8,9,10] --root --cont
cp Layers_3_${Name}.json Layers_4_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_4_${Name}.json  electrons_n1.feather -d -n ${Nseq}  ${random} ${activation} --freeze [4,5,6,7,8,9,10] --root --cont
cp Layers_4_${Name}.json Layers_5_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_5_${Name}.json  electrons_n1.feather -d -n ${Nseq}  ${random} ${activation} --freeze [5,6,7,8,9,10] --root --cont
cp Layers_5_${Name}.json Layers_6_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_6_${Name}.json  electrons_n1.feather -d -n ${Nseq}  ${random} ${activation} --freeze [6,7,8,9,10]  --root --cont
cp Layers_6_${Name}.json Layers_7_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_7_${Name}.json  electrons_n1.feather -d -n ${Nseq} ${random} ${activation} --freeze [7,8,9,10]  --root --cont
cp Layers_7_${Name}.json Layers_8_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_8_${Name}.json  electrons_n1.feather -d -n ${Nseq} ${random} ${activation} --freeze [8,9,10]  --root --cont
cp Layers_8_${Name}.json Layers_9_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_9_${Name}.json  electrons_n1.feather -d -n ${Nseq} ${random} ${activation} --freeze [9,10]  --root --cont
cp Layers_9_${Name}.json Layers_10_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_10_${Name}.json  electrons_n1.feather -d -n ${Nseq} ${random} ${activation} --freeze [10]  --root --cont
cp Layers_10_${Name}.json Layers_11_${Name}.json
../MLTrainer_exy.py -m 6 --rate 1.e-6 --file Layers_11_${Name}.json  electrons_n1.feather -d -n $((Nseq*4))  -cp ${Nseq} ${random} ${activation} --root --cont

