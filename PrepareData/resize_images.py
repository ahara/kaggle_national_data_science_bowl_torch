import os
import sys
import subprocess


rfi = '/home/adam/data/bowl/raw/'
rfo = '/home/adam/data/bowl/resized_proportion_64x64/'

cmd = "convert %s -resize 64x64 -size 64x64 xc:white +swap -gravity center -composite %s"

fi = rfi + 'train/'
fo = rfo + 'train/'
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(fi + cls)
    for img in imgs:
        os.system(cmd % (fi + cls + '/' + img, fo + cls + '/' + img))


fi = rfi + 'test/'
fo = rfo + 'test/'

imgs = os.listdir(fi)
for img in imgs:
    os.system(cmd % (fi + img, fo + img))
