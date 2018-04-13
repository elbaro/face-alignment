import tinder
tinder.setup(parse_args=False)

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import models_tensorflow
import cv2
import numpy as np

import torch  # torch for matrix product
def transform(point, center, scale, resolution, invert=False):
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

# center: float,float]
# scale: float
def crop(image, center, scale, resolution=256.0):
    # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg



# input: 1x68x64x64
# center : 2
# scale : scalar
# ret1 : 1x68x2 (coord in face img)
# ret2 : 1x68x2 (coord in orig img)
def get_preds_fromhm(hm, centers=None, scales=None):

    hm = torch.from_numpy(hm)
    hm = hm.permute(0,3,1,2).contiguous()

    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())

    # batch mode
    for i in range(hm.size(0)):  # batch index
        for j in range(hm.size(1)):  # 68
            preds_orig[i, j] = transform(preds[i, j], centers[i], scales[i], hm.size(2), True)

    return preds, preds_orig


class Detector(object):
    def __init__(self):

        # use JIT XLA: 2.6 FPS -> 3.0 FPS
        config = tf.ConfigProto()
        # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        self.sess = tf.Session(config=config)

        # use model code
        self.img = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='img')
        self.outs = models_tensorflow.FAN(self.img)
        saver = tf.train.Saver()
        saver.restore(self.sess, 'tf/weights.ckpt')

    # img condition:
    # given detect_box = [y,x,h,w],
    # let center.x (float) = middle point.x
    # let center.y (float) = middle point.y - height*0.12 (upward)
    # let scale = (width + height) / 195.0

    def detect(self, img_batch:np.ndarray):
        # img = img.reshape((1,)+img.shape) * 1.0
        outs = self.sess.run(self.outs, feed_dict={self.img: img_batch})
        return outs


def main():
    fan = Detector()

    img = Image.open('/home/emppu/Pictures/face.png')
    if img.mode!='RGB':
        img = img.convert('RGB')
    img = np.array(img)*1.0
    # img = cv2.resize(img, (256, 256))


    # y,x,height,width = [0, 0, 256, 256]
    y,x,height,width = [214.51, 144.209, 428.884, 287.85083]
    center = (x+width/2, y+height/2-height*0.12)
    scale = (width + height) / 195.0

    inp = crop(img, center, scale)
    batch = [inp]
    centers = [center]
    scales = [scale]

    batch = np.stack(batch, axis=0).astype(np.float32)
    batch /= 255.0
    batch_size = batch.shape[0]

    print('inp sum %.6f' %  batch.sum())
    out = fan.detect(batch)
    print('out sum %.6f' % out[-1].sum())

    pts, pts_img = get_preds_fromhm(out[-1], centers, scales)
    pts, pts_img = pts.view(batch_size, 68, 2) * 4, pts_img.view(batch_size, 68, 2)

    img = img.astype(np.uint8)

    for p in pts_img[0]:
        cv2.circle(img, (int(p[0]),int(p[1])),2,(255,0,0),-1)

    Image.fromarray(img).show()

main()
