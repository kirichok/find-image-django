from __future__ import division
import numpy as np
import cv2
# from matplotlib import pyplot as plt
import urllib
import math
from os.path import basename, splitext
import cPickle
import zlib
from operator import itemgetter

KP_EXT = '.kp'
DES_EXT = '.png'


def loadImageFromUrl(url, color=cv2.IMREAD_GRAYSCALE, resize=True, maxSize=800):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, color)
    if resize:
        return __getDeltaTransformation(image, maxSize)
    else:
        return image
        # return resize and __getDeltaTransformation(image, maxSize) or image


def loadImageFromPath(path, color=cv2.IMREAD_GRAYSCALE, resize=True, maxSize=800):
    image = cv2.imread(path, color)
    if resize:
        return __getDeltaTransformation(image, maxSize)
    else:
        return image
        # return resize and __getDeltaTransformation(image, maxSize) or image


def __getDeltaTransformation(image, maxSize):
    height, width = image.shape[:2]
    maxValue = max(height, width)
    delta = 1
    if maxValue > maxSize:
        delta = maxSize / maxValue
    return cv2.resize(image, (math.trunc(width * delta), math.trunc(height * delta)), interpolation=cv2.INTER_CUBIC)


def keypointDesCalc(image, savePath=''):
    kp, des = sift.detectAndCompute(image, None)
    if savePath:
        saveKpDesToPath(kp, des, savePath + KP_EXT)
        # saveKeypointToPath(kp, savePath + KP_EXT)
        # saveDesToPath(des, savePath + DES_EXT)
    return kp, des


def loadKeypointFromPath(path):
    index = cPickle.loads(open(path).read())
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


def loadKpDesFromPath(path, decompress=True):
    if decompress:
        index = cPickle.loads(zlib.decompress(open(path).read()))
    else:
        index = cPickle.loads(open(path).read())
    kp = []
    des = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
        des.append(point[6])
    return kp, np.array(des)


def loadDesFromPath(path):
    return loadImageFromPath(path, cv2.IMREAD_COLOR, False)


def saveKeypointToPath(kp, path):
    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        index.append(temp)
    f = open(path, "wb")
    f.write(cPickle.dumps(index))
    f.close()


def saveKpDesToPath(kp, des, path, compress=True):
    index = []
    i = 0
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, des[i])
        index.append(temp)
        i += 1
    f = open(path, "w")
    if compress:
        f.write(zlib.compress(cPickle.dumps(index)))
    else:
        f.write(cPickle.dumps(index))
    f.close()


def pack(kp, des, name, compress=True):
    data = {'f': name, 'k': [], 'd': des}
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        data['k'].append(temp)

    data = cPickle.dumps(data)
    if compress:
        data = zlib.compress(data)
    return data


def unpack(data, decompress=True):
    if decompress:
        data = cPickle.loads(zlib.decompress(data))
    else:
        data = cPickle.loads(data)
    kp = []
    des = data['d']
    name = data['f']
    for point in data['k']:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return name, kp, np.array(des)


def loadKps(path):
    return cPickle.loads(zlib.decompress(open(path).read()))


def saveKps(data, path):
    f = open(path, "wb")
    f.write(zlib.compress(cPickle.dumps(data)))


def saveDesToPath(des, path):
    cv2.imwrite(path, des)


def match(des1, des2):
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    return good


def getKpDes(img):
    return sift.detectAndCompute(img, None)


def sortKp(kp, des, count):
    _kp = []
    i = 0
    for point in kp:
        _kp.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id, des[i]))
        i += 1

    _kp = sorted(_kp, key=itemgetter(1), reverse=True)

    r_kp = []
    r_des = []
    for point in _kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        r_des.append(point[6])
        r_kp.append(temp)
        if len(r_kp) == count:
            break

    return r_kp, np.asarray(r_des, np.float32)


def compare(name1, name2, img1, img2):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        cv2.imwrite("./matched/%d %s_%s" % (len(good), name1, name2), img3)
        print "Matched (%s - %s) - %d/%d" % (name1, name2, len(good), MIN_MATCH_COUNT)
    else:
        print "Not enough matches are found (%s - %s) - %d/%d" % (name1, name2, len(good), MIN_MATCH_COUNT)


def fileName(str):
    return splitext(basename(str))[0]


MIN_MATCH_COUNT = 300

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# e1 = cv2.getTickCount()
# e2 = cv2.getTickCount()
# time = (e2 - e1) / cv2.getTickFrequency()
# print "Time: %s" % (time)
