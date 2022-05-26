# Official implementation of LIMES 
### (<ins>Li</ins>ghtweight Conditional <ins>M</ins>odel <ins>E</ins>xtrapolation for <ins>S</ins>treaming Data under Class-Prior Shift)

The LIMES method deals with a problem of Class-Prior Shift in continual learning. 
It incorporates bias correction term where extrapolation of class distribution is used. 
To learn more about the LIMES, see the paper by Paulina Tomaszewska and Christoph H. Lampert (tbd).

### Setup
We recommend creating a new conda virtual environment:
```
conda create -n LIMES python=3.8 -y
conda activate LIMES
pip install -r requirements.txt
```

### Usage
The data used in the experiements described in the paper is provided in the dehydrated form at the website: https://cvml.ist.ac.at/geo-tweets/
Raw tweets cannot be released due to Twitter Streaming API's Terms of Service. 
The data can be rehydrated using Twarc tool (https://github.com/DocNow/twarc) with just single line of code.

Later, the scripts should be used in the following order:
- process_raw_data.py (can be run using run_process_raw_data.sh on SLURM queuing system, for increase efficiency processing of each file with data can be a separate job)
- embeddings.py (can be run using analogical script to run_process_raw_data.sh)
- subsample_realizations.py
- training.py (can be run using run_training.sh)
 
For further details on the implementation and the instructions on code usage, see the companion paper (tbd).


This project is under the MIT license. See LICENSE for details.


