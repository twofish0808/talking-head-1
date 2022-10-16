# ntut-talking-head
## 1.The project
### 1.Face_rotator train
#### (1)Run the renamer.py, remenber to change the path in the renamer.py
#### (2)Use the new rotater and combiner to train the model, the model is the same as the old way, so you can use your own .pt file to keep training.

### 2.Schedule
#### 2021.08.17 由學長畢業論文交接。Start training Face_Morpher.
#### 2021.10.16 Start traing Rotator(by using L1loss). We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.10.29 Start traing Combiner(by using L1Loss).We use about 20 models by training 2 days to check whether the traing progress is OK.
#### 2021.11.15 Start traing Rotator(by using L1loss). Full model train.
#### 2021.12.20 Switch the Rotator to perceptual loss. With full model train.
#### 2021.12.26 Try to add noise on the training data.
#### 2022.03.19 Add our own app, with adjustable brightness to improve user's experience.

### 3.Run Demo
#### Replace the app folder, and put test_app folder in the root directory,and create test and checkpoint those two floders. Put your .pt file in the checkpoint folder. It should be like this.
```
	+--Talking-head-anime
	 |-test_app
	 |-test
	 |
	 +--app
	 | |-demo.py
	 | ∟poser_test.py
	 |
	 +--checkpoint
	   |-combiner.pt
	   |-face_morpher.pt
	   ∟two_algo_face_rotator.pt
```
#### And you can run like this.
```
   $python app/poser_test.py
```
#### Also the puppeteer.
##### To install enviroment
````
   $conda env create -f demo.yml
````
##### Run puppeteer.
````
   $python app/demo.py
````

### 4.enviroment
#### matplotlib(all the version is fine)
#### pytorch==1.8 or above(includes all the dependency packages)
#### python==3.6 or above(better use python 3.9.7)
#### cuda==10.2 or above (better use cuda11.1 or above)
#### We suggest you use the RTX2060 or above, or it will not be smooth in the demo.




   
