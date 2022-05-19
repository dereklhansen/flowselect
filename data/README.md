Download instructions for RNAsq and soybean data:

# RNAsq
-  Install ```scvi-tools``` and ```scanpy``` in a separate virtualenv. This is because ```scvi-tools``` relies on a version of pytorch-lightning which is incompatible with the bundled DDLK code.
-  Run ```get_scrnaseq_data.py``` inside the ```data/``` directory.

# Soybean 
The exact files for Soybean data can be downloaded here: ```https://github.com/kateyliu/DL_gwas```. Place each file in "./data/soybeans" directly. For example, "OIL/QA_oil.txt" should be placed in "./data/soybeans/QA_oil.txt".
