import sys


class MinHeap:

    def __init__(self, maxsize):
        self.maxSize = maxsize
        self.currentSize = 0
        self.Heap = [0] * (self.maxSize + 1)
        self.Heap[0] = -1 * sys.maxsize #lowest value
        self.values = [0] * (self.maxSize + 1)
        self.FRONT = 1

    # Function to return the position of
    # parent for the node currently
    # at pos
    def parent(self, pos):
        return pos // 2

    # Function to return the position of
    # the left child for the node currently
    # at pos
    def leftChild(self, pos):
        return 2 * pos

    # Function to return the position of
    # the right child for the node currently
    # at pos
    def rightChild(self, pos):
        return (2 * pos) + 1

    # Function that returns true if the passed
    # node is a leaf node
    def isLeaf(self, pos):
        if pos >= (self.currentSize // 2) and pos <= self.currentSize:
            return True
        return False

    # Function to swap two nodes of the heap
    def swap(self, fpos, spos):
        self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos]
        self.values[fpos], self.values[spos] = self.values[spos], self.values[fpos]

    # Function to heapify the node at pos
    def minHeapify(self, pos):

        # If the node is a non-leaf node and greater
        # than any of its child
        if not self.isLeaf(pos):
            if (self.Heap[pos] > self.Heap[self.leftChild(pos)] or
                    self.Heap[pos] > self.Heap[self.rightChild(pos)]):

                # Swap with the left child and heapify
                # the left child
                if self.Heap[self.leftChild(pos)] < self.Heap[self.rightChild(pos)]:
                    self.swap(pos, self.leftChild(pos))
                    self.minHeapify(self.leftChild(pos))

                # Swap with the right child and heapify
                # the right child
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.minHeapify(self.rightChild(pos))

    # Function to insert a node into the heap
    def insert(self, elementScore,elementValue):
        if self.currentSize >= self.maxSize:
            self.Heap.append(0)
            self.values.append(0)
            self.maxSize += 1
        self.currentSize += 1
        self.Heap[self.currentSize] = elementScore
        self.values[self.currentSize] = elementValue
        current = self.currentSize

        while self.Heap[current] < self.Heap[self.parent(current)]:
            self.swap(current, self.parent(current))
            current = self.parent(current)

    # Function to print the contents of the heap
    def Print(self):
        for i in range(1, (self.currentSize // 2) + 1):
            print(" PARENT : " + str(self.Heap[i]) + " LEFT CHILD : " +
                  str(self.Heap[2 * i]) + " RIGHT CHILD : " +
                  str(self.Heap[2 * i + 1]))

    # Function to build the min heap using
    # the minHeapify function
    def minHeap(self):

        for pos in range(self.currentSize // 2, 0, -1):
            self.minHeapify(pos)

    # Function to remove and return the minimum
    # element from the heap
    def remove(self):

        poppedScore = self.Heap[self.FRONT]
        poppedValue = self.values[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.currentSize]
        self.values[self.FRONT] = self.values[self.currentSize]
        self.currentSize -= 1
        self.minHeapify(self.FRONT)#heapify again so min goes to top again
        return poppedValue


