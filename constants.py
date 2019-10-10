#coding:utf-8

#pulic
BASE_DIR = "/home/dengerqiang/Documents/WORK/"
VQA_BASE = BASE_DIR + 'VQA/'
DATA10 = VQA_BASE + "data1.0/"
DATA20 = VQA_BASE + "data2.0/"
IMAGE_DIR = VQA_BASE + "images/"
GLOVE_DIR = BASE_DIR + 'glove/'
NLTK_DIR = BASE_DIR + "nltk_data"

#private
WORK_DIR = BASE_DIR+ "VQA/DenseCoAttention/"
GLOVE_FILE = "glove.6B.100d.pt"


#global arguments
RNN_DIM = 100
CNN_TYPE = "resnet152"