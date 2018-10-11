# CNN_Compression (PyTorch Implementation)
Code repository for our paper "Coreset-Based Neural Network Compression", published in ECCV 2018 

> [Moitreya Chatterjee*, Abhimanyu Dubey*, and Narendra Ahuja, "Coreset-Based Neural Network Compression. ", ECCV (2018)](https://arxiv.org/pdf/1807.09810) [* - Equal Contribution]

## Dependencies
- Python 2.7
- Pytorch 0.4.0
- Anaconda 3.6

We essentially run 4 files in sequence. These described in more details below:

# Pruning the network
Run the following in the specified order:
CNN_NList.py, followed by Gen_Clust_New_AllLayers_Omp.py, and finally Val_Unsup_ModNet_AlexNet.py

Please make sure to adjust the thresholds appropriately for Gen_Clust_New_AllLayers_Omp.py, and Val_Unsup_ModNet_AlexNet.py to induce an appropriate level of pruning.

Following includes the instructions in greater detail:
# CNN_Nlist.py 
- Extracts activations over the training set. Please set the path and file names of the output files
```
python CNN_Nlist.py -e -b 4 -j 2 --pretrained -a alexnet <path to dataset>
```
--  The Dataset Folder, should have subfolders 'val' and 'train', the -e flag indicates we are running in evaluation mode, i.e. we are not re-training the model, -b 4 is the batch size, -j 2 (preferably leave this unchanged) is the number of workers responsible for creating the dataset loader, -a alexnet is the architecture of the model, --pretrained suggests we are to use the features of the pretrained model available in the model zoo.

# Gen_Clust_New_AllLayers_Omp.py 
- Creates a mask determining which filters to prune and which to retain. Please set the path and file names of the output files and the thresholds which determine the extent of pruning
```
python Gen_Clust_New_AllLayers_Omp.py
```


# Val_Unsup_ModNet_New.py 
- Modifies the network depending on the extent of compression. Please set the path and file names of the input files (i.e. the masks) and choose the appropriate threshold level for pruning

```
python Val_Unsup_ModNet_New.py -e -b 4 -j 2 --pretrained -a alexnet <path to dataset>
```
-- The Dataset Folder, should have subfolders 'val' and 'train', the -e flag indicates we are running in evaluation mode, i.e. we are not re-training the model, -b 4 is the batch size, -j 2 (preferably leave this unchanged) is the number of workers responsible for creating the dataset loader, -a alexnet is the architecture of the model, --pretrained suggests we are to use the features of the pretrained model available in the model zoo.

Following this, please run the Coreset compression algorithm as follows
# compress_pytorch.py 
- Allows for further compression using Coreset-S or Coreset-K. Set the paths to the input model and the output model files, set the compression thresholds for each layer and then run
```
python compress_pytorch.py -i /path/to/input/model -o /path/to/output/model -r 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 -t 2
```
-- The -t option reflects that which coreset compression model we want to use, 2 stands for Coreset-K, 1 stands for Coreset-S and so on. The -r option reflects the compression ratios at the different layers. The -i option is the path to the input model, the -o option is the path to the output model.

If you find this repo useful, please cite the following paper
## Citation
```
@inproceedings{dubey2018coreset,
  title={Coreset-Based Neural Network Compression},
  author={Dubey, Abhimanyu and Chatterjee, Moitreya and Ahuja, Narendra},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={454--470},
  year={2018}
}
```
