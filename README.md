Visual Entity Linking
-----------------------------------------------------------------
Environment Setup
-----------------------------------------------------------------
We have used torch for our project. So here are the instruction to install torch:

To install:
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

To uninstall:
rm -rf ~/torch

We have used following packages or torch:
1) image : install it using 
$luarocks install image

2) nn : install it using
$luarocks install nn

3) optim : install it using
$luarocks install optim
-----------------------------------------------------------------
Dataset Requirement
-----------------------------------------------------------------
To excute our project, you need to down visual genome dataset.
The homepage for visual genome dataset is https://visualgenome.org/api/v0/api_home.html

You need to download following zip files from above link:
1) image.zip
2) image2.zip
3) image_data.zip
4) objects.zip
5) relations.zip

Unzip all the compressed files using following command:
unzip zipfilename.zip

move all the folder to where the source code is present

-----------------------------------------------------------------
Excute
-----------------------------------------------------------------
1) training the network :
To run the training of neural network use following command
$ sh train.sh

2) teseting the network :
To test the neural network use following command
$sh test.sh
Note: You need to modify the checkpoint number and test_model, if you have changed it in train.sh
