
### <h1 align="center" id="title"> 2Dto2D-DLE training example </h1>

1) Go to dle/data and make sure you have dowlonaded the minimal datset following the README

2) Link the code *cp ../../src/dle.py .* or *export PYTHONPATH=../../src*

3) Make sure you have set-up the correct python environment (e.g. conda activate dle)

4) Run 2Dto2D-DLE with 

	python  -c "from dle import dle ; dle = dle() ; dle.run()" \
		--usegpu True \
		--network 'cnn' \
		--dataset 'cfsflow_100' \
		--data_dir '../../data/' \
		--maptype 'f2' \
		--epochs 50 \
		--results_dir ''
		
5) At training, you may check the usage of your GPU with *watch -d -n 0.5 nvidia-smi*

6) A folder is created with the results of the training (model and evaluation items).

7) You may explore options with *--help*
