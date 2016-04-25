Bayesian Analysis of Stackexchange data
=====================

Bayesian analysis is applied to model the response time for questions on Stackexchange websites.

### Dependencies
p7zip used in `extract.sh` to extract .7z files in Stackexchange dump. Any other package can be used as long as the command with `7za` in `extract.sh` is appropriately modified.

### Extracting compressed Stackexchange dump
Run `extract.sh` in the project directory to extract stackexchange dump. Individual .7z files are assumed to be located in `stackexchange` folder in the project directory, and the extracted files are stored in `extracted` folder. The `unpacked` directory, including the directory structure are automatically created by the script. Additionally, it copies the directory structure of `unpacked` folder to `data` folder, for use by `parser.py`. The directory names can be changed in `extract.sh`

### Generating csv files
Run `parser.py` to generate csv files containing relevant attributes. The generated csv files are stored in `data` directory. The directory name can be changed in `extract.sh` and `parser.py`