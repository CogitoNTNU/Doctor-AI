=============================
Doctor AI
=============================
Deep learning for medical segmentation and classification

Spring 2023, Cogito NTNU.



In this project we use convolutional neural networks (CNNs) for computer vision. Using the U-Net architecture (Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015) as a basis, we tackle both classification-problems and segmentation-problems in the medical field.


.. image:: http://img.shields.io/badge/arXiv-1505.04597-orange.svg?style=flat
        :target: https://arxiv.org/abs/1505.04597
 Using the U-Net architecture.



=============================
Classification
=============================

Using the U-Net arcitechure we perform binary classification of 


Dataset: `Labeled Chest X-ray <https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images>`_ .






=============================
Segmentation
=============================


Segmentation of lungs 

.. image:: https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
        :target: https://colab.research.google.com/drive/13rYYCR1I8_mllIfTVtwQoyZmNruqBWPe?usp=sharing


Dataset: `LGG_MRI segmentation <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>`_ .


We performed lung segmentation achieving a F1 score of 0.957 and a IoU/jaccard score of 0.918.


.. image:: https://github.com/CogitoNTNU/Doctor-AI/raw/main/docs/images/summary-lung-seg-small.png 



Segmentation of Pneumothorax 

.. image:: https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
        :target: https://colab.research.google.com/drive/1ZEv1R5CZu4N7X9mqbrx9lB07_JyYt4UN?usp=sharing



Dataset: https://www.kaggle.com/datasets/vbookshelf/pneumothorax-chest-xray-images-and-masks




=============================
Technology used
=============================



=============================
Citation
=============================


As you use **Doctor-AI** for your own use, please cite the authors of the package::


	@article{doctorai2023,
	  title={Doctor AI - U-Net for medical segmentation and classification},
	  author={Vilhjalmurson, Vilhjalmur and Myhre, Sveinung and Bohne, Erik and Constantinos, Joel and Zhao, Ine},
	  year={2023},
	  publisher={Cogito NTNU}
	}


=============================
License
=============================
This project is licensed under the MIT License. See the LICENSE file for more information.
