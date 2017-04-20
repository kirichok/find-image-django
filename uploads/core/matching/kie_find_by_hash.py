import Queue
import threading
import cv2
import glob
import numpy as np
import kie_image as image
import argparse
import os
import time

HASH_PATH = '../images/hash/'
DES_EXT = '.des'

files = []
threading.stack_size(64*1024)

class FilesThread(threading.Thread):
    def __init__(self, threadID, name, queue, event, lock, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.queue = queue
        self.event = event
        self.lock = lock
        self.files = files

    def run(self):
        # print "Starting " + self.name
        while not self.event.is_set():
            self.lock.acquire()
            if not self.queue.empty():
                path = self.queue.get()
                self.lock.release()

                # t1 = cv2.getTickCount()


                des2 = image.loadDesFromPath(path)
                # des2 = np.asarray(image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False), np.float32)
                name = image.fileName(path)
                # t2 = cv2.getTickCount()

                if len(des2) >= 2:
                    self.lock.acquire()
                    self.files.append({'n': name, 'd': des2})
                    self.lock.release()

                    # t3 = cv2.getTickCount()
                    # time = (e2 - e1) / cv2.getTickFrequency()
                    # print "R: %s s P: %s s" % ((t2 - t1) / cv2.getTickFrequency(), (t3 - t2) / cv2.getTickFrequency())
            else:
                self.lock.release()
                # print "Exiting " + self.name


class RamThread(threading.Thread):
    def __init__(self, threadID, name, event, lock, workQueue, kp, des, finds):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.workQueue = workQueue
        self.lock = lock
        self.kp = kp
        self.des = des
        self.finds = finds
        self.event = event

    def run(self):
        while not self.event.is_set():
            self.lock.acquire()
            if not self.workQueue.empty():
                i = self.workQueue.get()
                data = files[i]
                self.lock.release()

                des2 = data['d']
                name = data['n']
                if len(des2) >= 2:
                    m = image.match(self.des, des2)
                    if len(m) >= 5:
                        self.lock.acquire()
                        self.finds.append({
                            'm': len(m),
                            'n': 'http://comicstore.cf/uploads/diamonds/%s.jpg' % image.fileName(name)})
                        self.lock.release()
                        print("Matched %s file %s" % (len(m), name))
            else:
                self.lock.release()


class HashThread(threading.Thread):
    def __init__(self, threadID, name, event, lock, workQueue, kp, des, finds):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.workQueue = workQueue
        self.lock = lock
        self.kp = kp
        self.des = des
        self.finds = finds
        self.event = event

    def run(self):
        i = 0
        print("Starting " + self.name)
        while not self.event.is_set():
            self.lock.acquire()
            if not self.workQueue.empty():
                path = self.workQueue.get()
                self.lock.release()
                i += 1
                des2 = image.loadImageFromPath(path, cv2.IMREAD_GRAYSCALE, False)
                des2 = np.asarray(des2, np.float32)
                if len(des2) >= 2:
                    name = image.fileName(path)
                    m = image.match(self.des, des2)
                    if len(m) >= 5:
                        self.lock.acquire()
                        self.finds.append({
                            'm': len(m),
                            'n': 'http://comicstore.cf/uploads/diamonds/%s.jpg' % image.fileName(name)})
                        self.lock.release()
                        print("Matched %s file %s" % (len(m), name))
                        # time.sleep(2)
            else:
                self.lock.release()

        print("Exiting %s - %d" % (self.name, i))


def loadHashFiles(hashPath=HASH_PATH, withSubFolders=True, threadsCount=50, ext=DES_EXT):
    print(len(files))
    if len(files) == 0:
        threadList = []
        count = 0
        while count < threadsCount:
            threadList.append("FilesThread-%d" % count)
            count += 1

        nameList = []
        if withSubFolders:
            folder = 0
            path = "%s%s/" % (hashPath, folder)
            while os.path.exists(path):
                for imagePath in glob.glob("%s*%s" % (path, ext)):
                    nameList.append(imagePath)
                    # if len(nameList) == 50000:
                    #     folder = -2
                    #     break
                folder += 1
                path = "%s%s/" % (hashPath, folder)
        else:
            for imagePath in glob.glob("%s*%s" % (hashPath, ext)):
                nameList.append(imagePath)
                # if len(nameList) == 50000:
                #     break

        if len(nameList) == 0:
            print("Hash files count is empty")
        else:
            print("Files count: %d " % len(nameList))

            event = threading.Event()
            queueLock = threading.Lock()
            workQueue = Queue.Queue(1000000)
            threads = []
            threadID = 1

            # Create new threads
            for tName in threadList:
                thread = FilesThread(threadID, tName, workQueue, event, queueLock, files)
                thread.daemon = True
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
            event.set()

            # Wait for all threads to complete
            for t in threads:
                t.join()


def checkFromRAM(imgPath, threadsCount=200):
    finds = []
    img = image.loadImageFromPath(imgPath, cv2.IMREAD_GRAYSCALE, False)
    kp, des = image.getKpDes(img)
    kp, des = image.sortKp(kp, des, 100)

    threadList = []
    count = 0
    while count < threadsCount:
        threadList.append("Thread-%d" % count)
        count += 1

    if len(files) == 0:
        print("Hash files count is empty")
    else:
        print("Files count: %d " % len(files))

        event = threading.Event()
        queueLock = threading.Lock()
        workQueue = Queue.Queue(len(files)+10)
        threads = []
        threadID = 1

        # Create new threads
        for tName in threadList:
            thread = RamThread(threadID, tName, event, queueLock, workQueue, kp, des, finds)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            threadID += 1

        # Fill the queue
        queueLock.acquire()
        for i in range(0, len(files)):
            workQueue.put(i)
        queueLock.release()

        t1 = cv2.getTickCount()
        # Wait for queue to empty
        while not workQueue.empty():
            pass

        # Notify threads it's time to exit
        event.set()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        t2 = cv2.getTickCount()
        time = (t2 - t1) / cv2.getTickFrequency()
        print("Time: %s s" % (time))

        return finds


def check(imgPath, hashPath=HASH_PATH, withSubFolders=True, threadsCount=200, ext=DES_EXT):
    finds = []
    img = image.loadImageFromPath(imgPath, cv2.IMREAD_GRAYSCALE, True, 200)
    kp, des = image.getKpDes(img)

    threadList = []
    count = 0
    while count < threadsCount:
        threadList.append("Thread-%d" % count)
        count += 1

    nameList = []
    if withSubFolders:
        folder = 0
        path = "%s%s/" % (hashPath, folder)
        print(path)
        while os.path.exists(path):
            for imagePath in glob.glob("%s*%s" % (path, ext)):
                nameList.append(imagePath)
            folder += 1
            path = "%s%s/" % (hashPath, folder)
    else:
        for imagePath in glob.glob("%s*%s" % (hashPath, ext)):
            nameList.append(imagePath)
            # if len(nameList) == 50000:
            #     break

    if len(nameList) == 0:
        print("Hash files count is empty")
    else:
        print("Files count: %d " % len(nameList))

        event = threading.Event()
        queueLock = threading.Lock()
        workQueue = Queue.Queue(1000000)
        threads = []
        threadID = 1

        # Create new threads
        for tName in threadList:
            thread = HashThread(threadID, tName, event, queueLock, workQueue, kp, des, finds)
            thread.daemon=True
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
        event.set()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return finds


if __name__ == '__main__':
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y" "1"):
            return True
        if v.lower() in ("no", "false", "f", "n" "0"):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threads", required=True, type=int, help="Threads count", default=50)
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-hf", "--hash", required=True, help="Path to the hash folder", default=HASH_PATH)
    ap.add_argument("-sf", "--subfolder", type=str2bool, help="Check sub folder", nargs='?', const=True)
    ap.add_argument("-ext", "--extention", help="Hash files extentions", default=DES_EXT)
    args = vars(ap.parse_args())

    e1 = cv2.getTickCount()
    # f = check(args['image'], args['hash'], args['subfolder'], args['threads'], args['extention'])
    loadHashFiles(args['hash'], args['subfolder'], args['threads'], args['extention'])
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("Time: %s s" % (time))
    # print("Finds: %s" % f)
    print("Exiting Main Thread")
