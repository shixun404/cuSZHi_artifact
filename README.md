# SC '25 cuSZ-Hi Artifact

This detached repo is based on the SC '25 artifacts for continous development.

## Artifact Setup

### Hardware

We require NVIDIA A100 GPU or newer to cover the essential functionality. In our paper, we evaluated the artifact on NVIDIA A100 (80GB) and RTX 6000 Ada to verify throughput scalability.

### Software

- We require an up-to-date mainstream Linux distro as the base environment.
  - e.g., CentOS 7 onward, Ubuntu 22.04.
- We require CUDA SDK of version 12.4 or 12.6.
  - corresponding to CUDA driver of version 550 onward.
- We require C++17-compliant host compiler.
  - e.g., GCC 9.3 onward.
- We require a modern cmake build system.
  - e.g., 3.18 onward. 


### Datasets/Inputs

The data setup will be done in setting up the workplace. 

<details>
<summary>
The details are folded here.
</summary>

- JHTDB 
  - Though hosted on https://turbulence.pha.jhu.edu/ as open data, it requires a token to access the data, which prohibits us from automating the data preprocessing. We can upload the data to a sharepoint if required.
- CESM, Miranda, Nyx, QMCPack 
  - hosted on https://sdrbench.github.io
- RTM data are from proprietary simulations
  - which are not open to the public.
  - We exclude the use of RTM in this artifact.

</details>  

### Setup Compilers

To use `module-load` to setup the toolchain:

```bash
## Please change the version accordingly.
module load cuda/12.4
module load gcc/9.3
````

<details>
<summary>
Alternative Compiler Setup using Spack 
</summary>

```bash
cd $HOME
git clone -c feature.manyFiles=true \
https://github.com/spack/spack.git
## Now, initialize Spack on terminal start
## It is recommended to add the next line to
## "$HOME/.bashrc" or "$HOME/.zshrc"
. $HOME/spack/share/spack/setup-env.sh
## For other shells, please refer to the
## instruction by typing (quotes not included)
## "$HOME/spack/bin/spack load"
spack compiler find
spack install gcc@9.3.0
spack install cuda@12.4.4%gcc@9.3.0

spack load gcc@9.3.0 cuda@12.4.4
export LD_LIBRARY_PATH=$(dirname $(which nvcc))/../lib64:$LD_LIBRARY_PATH
```

</details>

### Setup: Workspace

```bash
## (1) get the artifacts repo
cd $HOME ## It can be anywhere.
git clone --recursive \
  https://github.com/shixun404/25_SC_cuSZHi_artifact.git \
  sc25cuSZHi
cd sc25cuSZHi

## (2) setup
source setup-all.sh 12 <WHERE_TO_PUT_DATA_DIRS>

## (!!) clear build cache without removing data
bash setup-all.sh purge

## (3) prepare the data
python setup-data-v2.py
```

## Artifact Execution

Navigate back to the workplace using `cd $WORKSPACE`. Then, run for each dataset.

<details>
<summary>
Unfold to see commands to run the fast experiments covering cuSZ-Hi only.
</summary>

```bash
## $DATAPATH is set in setup-all.sh
## Please copy-paste each text block to run the per-dataset experiments.

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 512 512 512 --cmp cuSZi

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 384 384 256 --cmp cuSZi

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 69 69 33120 --cmp cuSZi
```
</details>


<details>

<summary>
Unfold to see commands to run the full experiments covering all compressors.
</summary>

```bash
## $DATAPATH is set in setup-all.sh
## Please copy-paste each text block to run the per-dataset experiments.

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 512 512 512

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 384 384 256

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_collection.py  \
  --input ${DATAPATH}/${THIS_DATADIR} \
  --output $DATAPATH/${THIS_DATADIR}_log \
  --dims 69 69 33120
```

</details>


## Artifact Analysis

<details>
<summary>
Unfold to see commands to analyze cuSZ-Hi only.
</summary>

```bash
## $DATAPATH is set in setup-all.sh
## Please copy-paste each text block to get the raw analysis results.

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 512 512 512 --cmp cuSZHi

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 384 384 256 --cmp cuSZHi

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 69 69 33120 --cmp cuSZHi
```
</details>

<details>
<summary>
Unfold to see commands to analyze all compressors.
</summary>

```bash
## $DATAPATH is set in setup-all.sh
## Please copy-paste each text block to get the raw analysis results.

## Nyx
THIS_DATADIR=SDRBENCH-EXASKY-NYX-512x512x512
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 512 512 512

## Miranda
THIS_DATADIR=SDRBENCH-Miranda-256x384x384
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 384 384 256

## QMC
THIS_DATADIR=SDRBENCH-SDRBENCH-QMCPack
python script_data_analysis.py  \
  --input ${DATAPATH}/${THIS_DATADIR}_log \
  --output $DATAPATH/${THIS_DATADIR}_csv \
  --dims 69 69 33120
```
</details>
