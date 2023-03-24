import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
# Additional library for image processing functions
import cv2 



'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
'''

MEAN_PATH = './'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = transforms.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)

# --------------------------------------------------------------------------------------------------------------------------------------------------------- 
# These are the definitions for the image processing types. To run on a specific type read the comment below. Additionally, Uncomment the transfromations of images below within the __getitem__ function and change to correct image processing function name

# Grayscale functions
# Uncomment the definiton below to run on the Intensity image processing

# def to_intensity(image):
#     gray = (1/3) * (image[0] + image[1] + image[2])
#     image = torch.stack([gray, gray, gray], dim=-1)
#     image[..., 0] = image[..., 0] * (1/3)
#     image[..., 1] = image[..., 1] * (1/3)
#     image[..., 2] = image[..., 2] * (1/3)
#     image = image.permute(2, 0, 1)
#     return image






# Uncomment the code below to run on the Luminance image processing

# def to_luminance(image):
#     gray =  0.3 * image[0] + 0.59 * image[1] + 0.11 * image[2]
#     image = torch.stack([gray, gray, gray], dim=-1)
#     image = image.permute(2, 0, 1)
#     return image






# Uncomment the code below to run on the Value image processing

# def to_value(image):
#     gray = image.max(dim=0)[0]
#     image = torch.stack([gray, gray, gray], dim=-1)
#     image = image.permute(2, 0, 1)
#     return image






# Uncomment the code below to run on the Luster image processing

# def to_luster(image):
#     max = image.max(dim=0)[0]
#     min = image.min(dim=0)[0]
#     gray = (max + min) / 2
#     image = torch.stack([gray, gray, gray], dim=-1)
#     image = image.permute(2, 0, 1)
#     return image





# Uncomment the code below to run on the Saravanan image processing

# def to_saravanan(image):
    # r = image[0]
    # b = image[1]
    # g = image[2]
    # y = (0.299 * b) + (0.587 * g) + (0.114 * b)
    # u  = (b - y) * 0.565
    # v =  (r - y) * 0.713
    # uv = u + v
    # r1 = r * 0.299
    # r2 = r * 0.587
    # r3 = r * 0.114
    # g1 = g * 0.299
    # g2 = g * 0.587
    # g3 = g * 0.114
    # b1 = b * 0.299
    # b2 = b * 0.587
    # b3 = b * 0.114
    # r4 = (r1 + r2 + r3) / 3
    # g4 = (g1 + g2 + g3) / 3
    # b4 = (b1 + b2 + b3) / 3
    # gray = (r4 + g4 + b4 + uv) / 4
    # image = torch.stack([gray, gray, gray], dim=-1)
    # image = image.permute(2, 0, 1)
    # return image






# Edge Detection
# Uncomment the definition below to run on the Canny image processing

# def to_canny(image):
#     image = transforms.functional.to_pil_image(image)
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(img, (5, 5), 3)
#     edges = cv2.Canny(image=blurred, threshold1=5, threshold2=25)
#     edges = torch.tensor(edges.astype(np.float32))
#     image = torch.stack([edges, edges, edges], dim=0)
#     image = image.permute(0, 1, 2)
#     return image





# Uncomment the code below to run on the Sobel image processing

# kernel_x = np.array([[-1, 0, 1],
#                      [-2, 0, 2],
#                      [-1, 0, 1]])

# kernel_y = np.array([[-1, -2, 1],
#                      [0, 0, 0],
#                      [1, 2, 1]])

# def to_sobel(image):
#     image = transforms.functional.to_pil_image(image)
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(img, (5, 5), 3)
#     sobel_x = cv2.filter2D(blurred, -1, kernel_x)
#     sobel_y = cv2.filter2D(blurred, -1, kernel_y)
#     edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
#     edges = torch.tensor(edges.astype(np.float32))
#     image = torch.stack([edges, edges, edges], dim=0)
#     image = image.permute(0, 1, 2)
#     return image





# Binary
# Uncomment the code below to run on the Otsu image processing

def to_otsu(image):
    image = transforms.functional.to_pil_image(image)
    image = image.convert('L')
    image_array = np.array(image)
    _, threshold = cv2.threshold(image_array , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarized_image = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    binarized_image = Image.fromarray(binarized_image)
    otsu = torch.tensor(np.array(binarized_image).astype(np.float32))
    image = torch.stack([otsu, otsu, otsu], dim=0)
    return image


# ------------------------------------------------------------------------------------------------------------------------------------------

class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metaFile = os.path.join(dataPath, 'metadata.mat')
        #metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])


        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:,0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid




    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)
            


        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

        # Adding participant ID to the metadata file
        participant = self.metadata['participant'][index]   

        # transforming images to the image processing type To run a specific type the function name must be changed to the corresponding function from above
        imFace = to_otsu(imFace)
        imEyeL = to_otsu(imEyeL)
        imEyeR = to_otsu(imEyeR)

    
        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        # returning all info including participant
        return row, imFace, imEyeL, imEyeR, faceGrid, gaze, participant
    
        
    def __len__(self):
        return len(self.indices)
