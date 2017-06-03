import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim as optim
import numpy as np
import scipy
import inception
from scipy import misc
from PIL import Image

MODEL_DIR = "./model/"
DATA_DIR = "./data/"
EPOCH = 10
BATCH_SIZE = 30
h = 299
w = 299
LUT = ["agriculture",
"artisinal_mine",
"bare_ground",
"blooming",
"blow_down",
"clear",
"cloudy",
"conventional_mine",
"cultivation",
"habitation",
"haze",
"partly_cloudy",
"primary",
"road",
"selective_logging",
"slash_burn",
"water"
]

models = os.listdir(MODEL_DIR)
now = 0
threshold = 0.5
if(len(models) == 0):
    mo = inception.inception_v3(pretrained = True).cuda()
else:
    now = 9
    mo = (torch.load(MODEL_DIR + "2_1000.pkl"))
    mo = mo.cuda()

class FineTuneModel(nn.Module):
    def __init__(self, num):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(incept.children())[:-1])
        self.classifier = nn.Sequential( nn.Linear(2048, num))
    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

#mo = FineTuneModel(17).cuda()
transformations = transforms.Compose([transforms.Scale((299, 299)), transforms.ToTensor()])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def import_image(s, path):
    name = s.strip()
    name = path + name
    img = Image.open(name)
    img = img.convert('RGB')
    #img = np.float32(np.swapaxes(np.swapaxes(img, 2, 3), 1, 2))
    img = transformations(img)
    return img
    
class Data(Dataset):
    def __init__(self, typ):
        f = open(DATA_DIR + typ + ".txt")
        self.typ = typ
        self.l = []
        cnt = 0
        for line in f.readlines():
            cnt += 1
            print cnt
            self.l.append(line)
        print(typ + ".txt read finish")
        np.random.shuffle(self.l)
    def __getitem__(self, i):
        path = DATA_DIR + self.typ + "/"
        image = import_image(self.l[i], path)
        return image, self.l[i]
    def __len__(self):
        return len(self.l)


#trainset = Data("train")
testset = Data("test")
test_loader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True)
print "start"
cnt = 0
print "switch to test"
out = open("summit.csv", "w")
out.write("image_name,tags\n")
for i, (data, names) in enumerate(test_loader):
    print i
    inputs = Variable(data).cuda()
    #outputs = mo.forward(inputs)
    output1, output2 = mo(inputs)
    outputs = 0.8 * output1 + 0.2 * output2
    outputs = outputs.data
    names = [x.split(".")[0] + "," for x in names]
    for i, f in enumerate(outputs):
        f = [sigmoid(x) for x in f]
        f = [ (1 if x > threshold else 0) for x in f]
        for j in xrange(len(f)):
            if(f[j] != 0):
                names[i] += LUT[j] + " "
    names = [x + "\n" for x in names]
    out.writelines(names)

out.close()
