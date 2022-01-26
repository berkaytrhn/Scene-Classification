# Scene-Classification
Scene Classification Using Tiny Image Features and Bag of Visual Words Together with KNN and SVM Classifiers



## Dataset
Scene Dataset: <a href="https://drive.google.com/file/d/1c_kTQNm-SXN6-ZmVcC8ZMOiLssRhy-1I/view">Link<a/>

## Requirements
	Modules:
		-cycler==0.10.0
		-joblib==1.0.1
		-kiwisolver==1.3.1
		-matplotlib==3.4.1
		-numpy==1.20.2
		-opencv-python==4.5.1.48
		-Pillow==8.2.0
		-pyparsing==2.4.7
		-python-dateutil==2.8.1
		-pytz==2021.1
		-scikit-learn==0.24.1
		-scipy==1.6.2
		-six==1.15.0
		-sklearn==0.0
		-threadpoolctl==2.1.0
	Python Version 3.8.0

## Running
	-Program directory has the created datasets(numpy arrays created by using most accurate parameters)
		because of running time concern(otherwise takes too long).
	-Should run program as "python main.py"
	-Program asks you to select one model among:
		->Bag of Visual Words With KNN
		->Bag of Visual Words With SVM
		->Tiny Image Features With KNN
		->Tiny Image Features With SVM
		and returns accuracy of selected model by using saved arrays.
	-Returns results about 2-3 seconds except "Tiny Image Features With SVM" because of max_iteration
		number of "LinearSVM" algorithm, it takes about 15-20 seconds.
	- Functions for testing on different parameters are commented initially, can be uncommented to run.
		BOVW-KNN -> Uncomment Lines 458,459
		BOVW-SVM -> Uncomment Lines 463,464
		TI-KNN   -> Uncomment Lines 468,469
		TI-SVM	 -> Uncomment Lines 473,474
## Extra 
	- To see confusion matrices, uncomment lines ->  337,338,285,286,228,229
	
