#!/bin/bash
for i in {1..20000}
do
    ./waf --run "scratch/NS3_Env.cc --RunNum=$(($i))"
done