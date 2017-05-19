import cv2

import glob
import os
import kie_image as im
import numpy as np

import threading
import Queue
import time

HASH_PATH = '../images/hash/'
DES_EXT = '.des'

index_params = dict(algorithm=1, trees=5)
search_params = {}
sift = cv2.xfeatures2d.SIFT_create()

# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,  # 20
#                     multi_probe_level=1)  # 2
#
# search_params = dict(checks=50)
# sift = cv2.ORB_create()



flanns = []
filenames = []


class LoadHashThread(threading.Thread):
    def __init__(self, threadID, name, task, flann, files, event):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.task = task
        self.files = files
        self.flann = flann
        self.event = event


    def run(self):
        while not self.event.is_set():
            if not self.task.empty():
                task = self.task.get()
                des = im.loadDesFromPath(task, 50)
                if len(des) >= 2:
                    self.files.append(im.fileName(task))
                    self.flann.add([des])


class checkHashThread(threading.Thread):
    def __init__(self, threadID, name, task, flann, files, event, lock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.task = task
        self.files = files
        self.flann = flann
        self.event = event
        self.lock = lock

    def run(self):
        while not self.event.is_set():
            if not self.task.empty():
                self.lock.acquire()
                task = self.task.get()
                self.lock.release()
                t_start = cv2.getTickCount()
                matches = self.flann.knnMatch(task, k=2)
                t_end = cv2.getTickCount()
                print "(%s) Time: %s" % (self.name, (t_end - t_start) / cv2.getTickFrequency())

                m = [m.imgIdx for m, n in matches if m.distance < n.distance * 1]
                if len(m) > 0:
                    results = {'v': [], 'c': []}
                    for i in m:
                        try:
                            index = results['v'].index(i)
                            results['c'][index] += 1
                        except ValueError:
                            results['v'].append(i)
                            results['c'].append(1)

                    print '%d:\n %s' % (len(m), ' '.join(
                        ['- %s(%s): %s\n' % (results['v'][i], results['c'][i], self.files[results['v'][i]]) for i in range(0, len(results['v']))]))
                else:
                    print 'Not found'
            else:
                time.sleep(2)


def loadHashFiles(filesInQueue=10000, hashPath=HASH_PATH, ext=DES_EXT):
    qFiles = []
    folder = 0
    path = "%s%s/" % (hashPath, folder)
    count = 0
    countAll = 0
    curQueue = Queue.Queue(filesInQueue)
    qFiles.append(curQueue)
    while os.path.exists(path):
        for imagePath in glob.glob("%s*%s" % (path, ext)):
            curQueue.put(imagePath)
            count += 1
            countAll += 1
            if count == filesInQueue:
                qFiles.append(curQueue)
                curQueue = Queue.Queue(filesInQueue)
                count = 0

            # if countAll == 1000:
            #     folder = -2
            #     break

        folder += 1
        path = "%s%s/" % (hashPath, folder)

    if countAll == 0:
        print("Hash files count is empty")
    else:
        print("Files count: %d " % countAll)

        threadList = []
        count = 0
        while count < len(qFiles):
            threadList.append("FilesThread-%d" % count)
            filenames.append([])
            count += 1

        event = threading.Event()
        threads = []
        threadID = 0

        # Create new threads
        for i, tName in enumerate(threadList):
            currflann = cv2.FlannBasedMatcher(index_params, search_params)
            flanns.append(currflann)
            thread = LoadHashThread(threadID, tName, qFiles[i], currflann, filenames[i], event)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            threadID += 1

        # Wait for queue to empty
        empty = False
        while not empty:
            empty = True
            for q in qFiles:
                if not (empty and q.empty()):
                    empty = False

        # Notify threads it's time to exit
        event.set()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        print('Start training ...')
        for f in flanns:
            f.train()
        print('Training end')


def startThreads(lock):
    threadList = []
    count = 0
    while count < len(flanns):
        threadList.append("CheckFilesThread-%d" % count)
        filenames.append([])
        count += 1

    event = threading.Event()
    threads = []
    threadID = 0
    tasks = []
    # Create new threads
    for i, tName in enumerate(threadList):
        task = Queue.Queue(100)
        tasks.append(task)
        thread = checkHashThread(threadID, tName, task, flanns[i], filenames[i], event, lock)
        thread.daemon = True
        thread.start()
        threads.append(thread)
        threadID += 1

    return threads, event, tasks


def stopThreads(threads, event):
    # Notify threads it's time to exit
    event.set()

    # Wait for all threads to complete
    for t in threads:
        t.join()


def checkFromRAM(path):
    img = im.loadImageFromPath(path, resize=True, maxSize=800)
    _kp, _des = sift.detectAndCompute(img, None)
    kp, des = im.sortKp(_kp, _des, 200)

    t_start = cv2.getTickCount()
    res = []
    for i, flann in enumerate(flanns):
        matches = flann.knnMatch(des, k=2)
        m = [m.imgIdx for m, n in matches if m.distance < n.distance * 0.6]

        if len(m) > 0:
            results = {'v': [], 'c': []}
            for j in m:
                try:
                    index = results['v'].index(j)
                    results['c'][index] += 1
                except ValueError:
                    results['v'].append(j)
                    results['c'].append(1)

            for j in range(0, len(results['v'])):
                res.append({
                    'm': results['c'][j],
                    'n': 'http://comicstore.cf/uploads/diamonds/%s.jpg' % filenames[i][results['v'][j]]})

            print '%d:\n %s' % (len(m), ' '.join(
                ['- %s(%s): %s\n' % (results['v'][j], results['c'][j], filenames[i][results['v'][j]])
                 for j in range(0, len(results['v']))]))
        else:
            print 'Not found'
    t_end = cv2.getTickCount()
    print "Time: %s" % ((t_end - t_start) / cv2.getTickFrequency())

    return res


def testFlan():
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    load_descriptors(flann)

    # img = im.loadImageFromUrl('http://comicstore.cf/uploads/diamonds/STK360887.jpg', resize=True, maxSize=200)
    img = im.loadImageFromPath('./images/z13.jpg', resize=True, maxSize=800)
    _kp, _des = sift.detectAndCompute(img, None)
    kp, des = im.sortKp(_kp, _des, 50)

    t_start = cv2.getTickCount()
    matches = flann.knnMatch(des, k=2)
    t_end = cv2.getTickCount()
    print "Time match: %s" % ((t_end - t_start) / cv2.getTickFrequency())
    m = [(m.queryIdx, m.trainIdx, m.imgIdx) for m, n in matches if m.distance < n.distance * 0.75]
    duples = [c for n, (a, b, c) in enumerate(m) if c in [cc for aa, bb, cc in m[:n if n > 0 else n + 1]]]
    no_duples = [c for n, (a, b, c) in enumerate(m) if c not in duples and c not in [cc for aa, bb, cc in m[:n]]]
    print "Points: %d,\n no duples(%d): %s,\n duples(%d): %s" % (len(m),
                                                                 len(no_duples), ' '.join([str(i) for i in no_duples]),
                                                                 len(duples), ' '.join([str(i) for i in duples]))


def load_descriptors(knn, hashPath=HASH_PATH, withSubFolders=True, ext=DES_EXT):
    nameList = []
    if withSubFolders:
        folder = 0
        path = "%s%s/" % (hashPath, folder)
        while os.path.exists(path):
            for imagePath in glob.glob("%s*%s" % (path, ext)):
                nameList.append(imagePath)
            folder += 1
            path = "%s%s/" % (hashPath, folder)
    else:
        for imagePath in glob.glob("%s*%s" % (hashPath, ext)):
            nameList.append(imagePath)

    if len(nameList) == 0:
        print("Hash files count is empty")
        return

    print("Files count: %d " % len(nameList))
    for i, n in enumerate(nameList[:10000]):
        if i % 10000 == 0:
            print '%d/50000' % i
        knn.add([im.loadDesFromPath(n)[:50]])
        # knn.train()
    knn.train()
    print("Loaded")


if __name__ == '__main__':
    loadHashFiles(120000)

    lock = threading.Lock()
    threads, event, tasks = startThreads(lock)

    inpt = -1
    while inpt != 0:
        try:
            inpt = int(raw_input(' 0 - exit \n 1 - check \n'))
        except ValueError:
            inpt = -1
            continue

        if inpt == 1:
            try:
                file_name = raw_input('Input filename: ')
            except ValueError:
                file_name = None
                continue

            img = im.loadImageFromPath('../images/%s' % file_name, resize=True, maxSize=800)
            _kp, _des = sift.detectAndCompute(img, None)
            kp, des = im.sortKp(_kp, _des, 200)

            lock.acquire()
            for t in tasks:
                t.put(des)
            lock.release()
            pass
        else:
            pass

    stopThreads(threads, event)
