from __future__ import print_function
import os
import glob
import dlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from enum import Enum
from skimage import io
import numpy as np

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        if self.enable_cuda or self.use_cnn_face_detector:
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignemnt networks
        self.face_alignemnt_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)

        self.face_alignemnt_net.load_state_dict(fan_weights)

        if self.enable_cuda:
            self.face_alignemnt_net.cuda()
        self.face_alignemnt_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda()
            self.depth_prediciton_net.eval()

    def detect_faces(self, image):
        """Run the dlib face detector over an image

        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.

        Returns:
            Returns a list of detected faces
        """

        # large? resize
        long_side = max(image.shape[0], image.shape[1])
        if long_side > 1024:
            scale = 1024/long_side
            resized = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            detects = self.face_detector(resized, 1)

            for d in detects:
                r = d.rect
                d.rect = dlib.rectangle(int(r.left()/scale), int(r.top()/scale),
                                        int(r.right()/scale), int(r.bottom()/scale))
                return [d]
            return []

        return self.face_detector(image, 1)

    def process_batch(self, input_images, y_x_height_width=None):
        assert type(input_images) == list
        batch = []
        centers = []
        scales = []
        face_found = []

        for i_img, input_image in enumerate(input_images):
            if isinstance(input_image, str):
                try:
                    image = io.imread(input_image)
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=2).repeat(3, axis=2)  # [H,W] -> [H,W,3]
                except IOError:
                    print("error opening file :: ", input_image)
                    return None
                input_images[i_img] = image
            else:
                image = input_image

            found = False
            if y_x_height_width is None:
                detected_faces = self.detect_faces(image)
                if len(detected_faces) > 0:
                    found = True
                    # find largest rect
                    mx = 0
                    r = None
                    for i, d in enumerate(detected_faces):
                        area = d.rect.area()
                        if area > mx:
                            mx = area
                            r = d.rect
                    center = torch.FloatTensor(
                        [r.right() - (r.right() - r.left()) / 2.0, r.bottom() -
                            (r.bottom() - r.top()) / 2.0])
                    center[1] = center[1] - (r.bottom() - r.top()) * 0.12
                    scale = (r.right() - r.left() + r.bottom() - r.top()) / 195.0
            else:
                found = True
                top, left, height, width = y_x_height_width[i_img]
                bottom = top + height
                right = left + width
                center = torch.FloatTensor(
                    [right - width / 2.0, bottom - height / 2.0])
                center[1] = center[1] - (bottom - top) * 0.12
                scale = (right - left + bottom - top) / 195.0

            face_found.append(found)
            if found:
                inp = crop(image, center, scale)
                batch.append(inp)
                centers.append(center)
                scales.append(scale)

        if len(batch) == 0:  # 0 face
            return face_found, []

        batch = np.stack(batch, axis=0)
        batch = np.transpose(batch, axes=(0, 3, 1, 2)).astype(np.float32) / 255.0
        batch = torch.from_numpy(batch).contiguous()
        batch_size = batch.size(0)
        scales = torch.FloatTensor(scales)

        if self.enable_cuda:
            batch = batch.cuda(async=True)

        batch_out = self.face_alignemnt_net(Variable(batch, volatile=True))[-1].data.cpu()
        if self.flip_input:
            batch_out += flip(self.face_alignemnt_net(Variable(flip(batch), volatile=True))
                              [-1].data.cpu(), is_label=True)

        pts, pts_img = get_preds_fromhm(batch_out, centers, scales)
        pts, pts_img = pts.view(batch_size, 68, 2) * 4, pts_img.view(batch_size, 68, 2)

        # 2D -> 3D
        heatmaps = np.zeros((batch_size, 68, 256, 256))
        for b in range(batch_size):
            for i in range(68):
                if pts[b, i, 0] > 0:
                    heatmaps[b, i] = draw_gaussian(heatmaps[b, i], pts[b, i], 2)
        heatmaps = torch.from_numpy(heatmaps).view(batch_size, 68, 256, 256).float()
        heatmaps = heatmaps.cuda(async=True)

        depth_pred = self.depth_prediciton_net(
            Variable(torch.cat((batch, heatmaps), 1), volatile=True)).data.cpu().view(batch_size, 68, 1)

        # scales : [B], depth_pred : [B, 68, 1]
        pts_img = torch.cat((pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scales.view(-1, 1, 1))))), 2)
        return face_found, pts_img.numpy()

    def process_folder(self, path, all_faces=False):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(glob.glob(os.path.join(path, files)))

        predictions = []
        for image_name in images_list:
            predictions.append(image_name, self.get_landmarks(image_name, all_faces))

        return predictions

    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)
