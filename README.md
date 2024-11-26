ZeROSearch
========

# overview

# setup
1. make conda virtual environment
   ```
   conda create -n py310 python=3.10
   ```
2. install packages
   ```
   conda activate py310
   pip install torch numpy pandas
   ```
# directory hierarchy
```
ZeROSearch
ㄴprofiledb
ㄴsrc
    ㄴzerosearch.py
    ㄴestimate.py
    ㄴpipe.py
    ㄴstage.py
    ㄴutils.py
    ㄴdevice_placement.py
    ㄴmodel_config.py
```
## profiledb
includes profiledb which is the execution time for each layer type of the given large language model. The layer types are embedding layer, transformer layer, post process layer.

# Usage

```
python zerosearch.py {arg1, arg2, ...}
```

## argument

## example


