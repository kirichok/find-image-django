import queue
import threading
import cv2
import glob
import numpy as np
import time
# import mysql_conn

import  uploads.core.matching.kie_image as image

import os, os.path
import argparse

HASH_PATH = '../images/hash/'
KP_EXT = '.kp'
DES_EXT = '.png'

exitFlag = 0

class HashThread(threading.Thread):
    def __init__(self, threadID, name, lock, workQueue, kp, des, finds):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.workQueue = workQueue
        self.lock = lock
        self.kp = kp
        self.des = des
        self.finds = finds

    def run(self):
        global exitFlag
        # print("Starting " + self.name)
        while not exitFlag:
            self.lock.acquire()
            if not self.workQueue.empty():
                path = self.workQueue.get()
                self.lock.release()

                des2 = image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False)
                des2 = np.asarray(des2, np.float32)
                if des2 and len(des2) >= 2:
                    name = image.fileName(path)
                    m = image.match(self.des, des2)
                    if len(m) >= 50:
                        self.lock.acquire()
                        self.finds.append({'m': len(m), 'n': name})
                        self.lock.release()
                        print("Matched %s file %s" % (len(m), name))
            else:
                self.lock.release()

        # print("Exiting " + self.name)


def check(imgPath, hashPath='/home/user/find-image-django/media/hash/', threadsCount=200):
    global exitFlag

    finds = []
    exitFlag = 0
    print('check: %s' % imgPath)
    img = image.loadImageFromUrl(imgPath, cv2.IMREAD_GRAYSCALE, True, 200)
    kp, des = image.getKpDes(img)

    threadList = []
    count = 0
    while count < threadsCount:
        threadList.append("Thread-%d" % count)
        count += 1

    nameList = []
    folder = 1
    path = "%s%s/" % (hashPath, folder)
    while os.path.exists(path):
        for imagePath in glob.glob("%s*%s" % (path, DES_EXT)):
            nameList.append(imagePath)
        folder += 1
        path = "%s%s/" % (hashPath, folder)

    if len(nameList) == 0:
        print("Files count is empty")

    else:
        print("Files count: %d " % len(nameList))

        queueLock = threading.Lock()
        workQueue = queue.Queue(1000000)
        threads = []
        threadID = 1

        # Create new threads
        for tName in threadList:
            thread = HashThread(threadID, tName, queueLock, workQueue, kp, des, finds)
            thread.start()
            threads.append(thread)
            threadID += 1

        # Fill the queue
        queueLock.acquire()
        for word in nameList:
            workQueue.put(word)
        queueLock.release()

        # Wait for queue to empty
        while not workQueue.empty():
            pass

        # Notify threads it's time to exit
        exitFlag = 1

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return finds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threads", required=True, type=int, help="Threads count", default=50)
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    e1 = cv2.getTickCount()
    f = check(args['image'], HASH_PATH, args['threads'])
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("Time: %s s" % (time))
    print("Finds: %s" % f)
    print("Exiting Main Thread")

    # img = image.loadImageFromPath(args['image'], cv2.IMREAD_GRAYSCALE, True, 200)
    # kp, des = image.getKpDes(img)
    #
    # threadList = []
    # count = 0
    # while count < args['threads']:
    #     threadList.append("Thread-%d" % count)
    #     count += 1
    #
    # nameList = []
    # folder = 1
    # path = "%s%s/" % (HASH_PATH, folder)
    # while os.path.exists(path):
    #     for imagePath in glob.glob("%s*%s" % (path, DES_EXT)):
    #         nameList.append(imagePath)
    #     folder += 1
    #     path = "%s%s/" % (HASH_PATH, folder)
    #     print(folder)
    #
    #
    # if len(nameList) == 0:
    #     print("Files count is empty")
    #     exit(0)
    # else:
    #     print("Files count: %d " % len(nameList))
    #
    # queueLock = threading.Lock()
    # workQueue = queue.Queue(1000000)
    # threads = []
    # threadID = 1
    #
    # # Create new threads
    # for tName in threadList:
    #     thread = HashThread(threadID, tName, workQueue)
    #     thread.start()
    #     threads.append(thread)
    #     threadID += 1
    #
    # # Fill the queue
    # queueLock.acquire()
    # for word in nameList:
    #     workQueue.put(word)
    # queueLock.release()
    #
    # e1 = cv2.getTickCount()
    # # Wait for queue to empty
    # while not workQueue.empty():
    #     pass
    #
    # # Notify threads it's time to exit
    # exitFlag = 1
    #
    # # Wait for all threads to complete
    # for t in threads:
    #     t.join()
    #
    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # print("Time: %s s" % (time))
    #
    # print("Exiting Main Thread")
