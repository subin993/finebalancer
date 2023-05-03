
NS3-SON README
================================

## OS and software preparation:

We base our experiment environment on Ubuntu 20.04 LTS and highly recommend that you do the same. This streamlines the setup process and avoids unexpected issues cause by incompatible software versions etc. Please make sure that you have Python installed. Also make sure that you have root or sudo permission.

This branch contains the entire ns-3 network simulator (ns-3.33) with ns3-gym (opengym) module.

## Install NS3-SON 

1. The first part of the preparation is to clone the repository:

```shell
git clone https://github.com/subin993/finebalancer.git
```

2. Next, install all dependencies required by ns-3.

```shell
apt-get install gcc g++ python python3-pip
```

3. Install ZMQ and Protocol Buffers libs:

```shell
sudo apt-get update
apt-get install libzmq5 libzmq3-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
```

4. Install Pytorch.

```shell
pip3 install torch
```

Following guideline of installation in https://pytorch.org

5. Building ns-3

```shell
./waf configure --enable-examples
./waf build
```

6. Install ns3-gym

```shell
pip3 install --user ./src/opengym/model/ns3gym
```

## Setting ns-3 environment

To run all of scenarios (NS3_Env.cc) in the directory of scratch, you can configure your own settings for scenarios (Number of UEs, Number of BSs, SrsPeriodicity, etc.) 

In the case of small scale scenario, make sure that the number of UEs are 40 and srsPeriodicity is equal to 80 ms.

In the case of large scale scenario, make sure that the number of UEs are 90 and srsPeriodicity is equal to 160 ms.

## Running ns-3 environment

To run all of the scenarios, open the terminal and run the command:

```shell
chmod +x ./FineBalancer.sh
./bash FineBalancer.sh
```

Note that, you don't have to repeat the following command after your first running.

```shell
chmod +x ./FineBalancer.sh
```

If you want to run only one episode, run the command:

```shell
./waf --run scratch/NS3_Env.cc
```

## Running agent of various algorithms

Open a new terminal and run the command:

```shell
cd ./scratch
python3 <name-of-algorithm>.py
```

Contact
================================
Subin Han, Korea University, subin993@korea.ac.kr
Eunsok Lee, Korea University, tinedge@korea.ac.kr


How to reference NS3-SON?
================================
Please use the following bibtex:

<blank>
