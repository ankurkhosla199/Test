# Modified version of final_code_symm2.py final_code_symm.py
# works for both apple and onions (setting the flag -> apple = True / False)
# Inputs single RGB image
# Define class object FruitShape(image)
# Use FruitShape().get_segmentation_mask() and FruitShape().get_contour() to kickstart the code
# width, height, final_image = FruitShape().get_axis_by_symmetry() -> gives raw symmetry axis

# OR call rotateImage(image) class, and get_axis_by_rotation() function
# It finds most optimal axis of symmetry, by rotating image around centroid point

import sys
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import statistics
from matplotlib import pyplot as plt
from sklearn import svm
import mahotas
import glob
import os
import pickle
from sklearn import model_selection
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class FruitShape:

    def __init__(self, image=None):
        """ Initializes the code
        @param image: 3-channel image of Fruit
        @param frame_threshold: Binary Mask of image
        @param combined: Black image on which contour and lines are drawn, to be returned later
        @param c: Boundary contour of fruit
        @param height, width: dimensions of prependicular line intercepted by the contour
        @param line_thickness: Thickness of lines to be drawn on combined
        """

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.image = image
        self.frame_threshold = None
        self.combined = None
        self.height = 0
        self.width = 0
        self.c = None
        self.line_thickness = 1
        self.centroid_flag = True
        self.apple = False
        self.centroid_x = 0
        self.centroid_y = 0
    
    def get_segmentation_mask(self):
        """ Image segemntation and provides mask of the image
        @param frame_HSV: Using HSV thresholds on original image
        """
        if self.apple == True:
            image_mean = cv2.pyrMeanShiftFiltering(self.image, 21, 51)
            frame_HSV = cv2.cvtColor(image_mean, cv2.COLOR_BGR2HSV)
            self.frame_threshold = cv2.inRange(frame_HSV, (0, 65, 0), (179, 255, 255), cv2.THRESH_BINARY)
            self.frame_threshold = cv2.medianBlur(self.frame_threshold, 5)
        else:
            image_temp = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            ret, self.frame_threshold = cv2.threshold(image_temp, 0, 255, cv2.THRESH_BINARY)
            self.frame_threshold = cv2.medianBlur(self.frame_threshold, 5)
    
    def get_contour(self):
        """ Intializes combined (black image) and draws boundary contour of the fruit
            defines centroid_x and centroid_y from the contour
        """

        cnts, _ = cv2.findContours(self.frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.c = max(cnts, key=cv2.contourArea)
        self.combined = np.zeros(self.frame_threshold.shape, dtype="uint8")
        cv2.drawContours(self.combined, self.c, -1, 255, self.line_thickness)
        M = cv2.moments(self.c)
        self.centroid_x = int(M['m10']/M['m00'])
        self.centroid_y = int(M['m01']/M['m00'])
    
    def very_close(self, a, b, tol=4.0):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol

    def S(self, si, sj, sigma=1):
        q = (-abs(si - sj)) / (sigma * (si + sj))
        return np.exp(q ** 2)

    def reisfeld(self, phi, phj, theta):
        return 1 - np.cos(phi + phj - 2 * theta)

    def midpoint(self, i, j):
        return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2

    def angle_with_x_axis(self, i, j):
        x, y = i[0] - j[0], i[1] - j[1]
        if x == 0:
            return np.pi / 2
        angle = np.arctan(y / x)
        if angle < 0:
            angle += np.pi
        return angle

    def superm2(self, image):
        """ Github Code to provide the points corresponding to line of symmetry
            Link: https://github.com/dramenti/symmetry
            Uses above 5 functions, and returns an array of [x,y] points
            Plots hexbin() of 'r vs theta' and choose co-ordinates of the max value
        @param image: Single channel image of Fruit
        """

        image2 = image.copy()
        if self.apple == True:
            mimage = np.fliplr(image)
        else:
            mimage = np.flipud(image)
        kp1, des1 = self.sift.detectAndCompute(image, None)
        kp2, des2 = self.sift.detectAndCompute(mimage, None)
        for p, mp in zip(kp1, kp2):
            p.angle = np.deg2rad(p.angle)
            mp.angle = np.deg2rad(mp.angle)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        houghr = np.zeros(len(matches))
        houghth = np.zeros(len(matches))
        weights = np.zeros(len(matches))
        i = 0
        good = []
        for match, match2 in matches:
            point = kp1[match.queryIdx]
            mirpoint = kp2[match.trainIdx]
            mirpoint2 = kp2[match2.trainIdx]
            mirpoint2.angle = np.pi - mirpoint2.angle
            mirpoint.angle = np.pi - mirpoint.angle
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi
            if mirpoint2.angle < 0.0:
                mirpoint2.angle += 2 * np.pi
            mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
            if self.very_close(point.pt, mirpoint.pt):
                mirpoint = mirpoint2
                good.append(match2)
            else:
                good.append(match)
            theta = self.angle_with_x_axis(point.pt, mirpoint.pt)
            xc, yc = self.midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)
            Mij = self.reisfeld(point.angle, mirpoint.angle, theta) * self.S(
                point.size, mirpoint.size
            )
            houghr[i] = r
            houghth[i] = theta
            weights[i] = Mij
            i += 1
        good = sorted(good, key=lambda x: x.distance)

        img3 = cv2.drawMatches(image, kp1, mimage, kp2, good[:15], None, flags=2)
        
        test = plt.hexbin(houghr, houghth, bins=200)
        co_ordinates = test.get_offsets()
        values = test.get_array()
        index = np.argmax(values)
        r = co_ordinates[index][0], 
        theta = co_ordinates[index][1]

        points = []
        if np.pi / 4 < theta < 3 * (np.pi / 4):
            for x in range(len(image2.T)):
                y = int((r - x * np.cos(theta)) / np.sin(theta))
                if 0 <= y < len(image2.T[x]):
                    points.append([y, x])
        else:
            for y in range(len(image2)):
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < len(image2[y]):
                    points.append([y, x])
        return points

    def get_line(self, theta, x, y, check=True):
        """ Draws a line, and part intercepted by the contour, returns the length of line intercepted by contour
        @param theta: Angle of line with x-axis, in radians
        @param x: X co-ordinate of the point
        @param y: Y co-ordinate of the point
        @param check: bool value used to denote if line intercepted by the contour should be drawn or not, true by default
        """

        ref = np.zeros(self.frame_threshold.shape, dtype="uint8")
        width = self.frame_threshold.shape[1]
        height = self.frame_threshold.shape[0]

        cv2.line(ref, (int(x - np.cos(theta)*width), int(y + np.sin(theta)*height)),
                    (int(x + np.cos(theta)*width), int(y - np.sin(theta)*height)), 
                    255, self.line_thickness)

        final_line = cv2.bitwise_and(self.frame_threshold, ref)
        n_white_pix = np.sum(final_line == 255)

        if check:
            self.combined = cv2.bitwise_or(final_line, self.combined)

        return n_white_pix / self.line_thickness

    def get_slope(self, points):
        """ Returns the slope of line, formed by the collection of (X, Y) co-ordinates
        @param points: array of (X, Y) co-ordinates
        """

        points = np.array(points)
        points.sort(axis=0)
        slope = np.arctan( (points[-1][0] - points[0][0]) / (points[-1][1] - points[0][1]) )
        slope = slope + np.pi
        return slope

    def get_symm_line_points(self):
        """ invokes superm2() funtion and returns the array of (X, Y) co-ordinates of symmetry axis
            input to superm2() function must be masked image, not the original image
        @param points: array of (X, Y) co-ordinates
        """

        temp_image = cv2.bitwise_and(self.image, self.image, mask=self.frame_threshold)
        temp_image = cv2.GaussianBlur(temp_image, (5, 5), 0)
        
        cv2.imwrite('test.jpg', temp_image)
        points = self.superm2(cv2.imread('test.jpg', 0))
        os.chmod('test.jpg' , 0o777)
        os.remove('test.jpg')
        points = np.array(points)
        return points

    def give_symmteric_axis(self):
        """ marks the symmetry axis of the fruit, inside the contour
            returns dimension of line segment and 2-D image of (contour + line segment)
        @param points: array of (X, Y) co-ordinates of symmetry axis
        @param height: dimension of line segment
        @param combined: 2-D image of contour and symmetry axis marked
        """

        points = self.get_symm_line_points()
        self.height = self.get_line(theta=self.get_slope(points), x=points[0][1], y=points[0][0])
        return self.height, self.combined 

    def give_symmteric_axis_centroid(self):
        """ marks the symmetry axis passing from centroid, inside the contour
            returns dimension of line segment and 2-D image of (contour + line segment)
        @param points: array of (X, Y) co-ordinates of symmetry axis
        @param height: dimension of line segment
        @param combined: 2-D image of contour and symmetry axis marked
        """

        points = self.get_symm_line_points()
        M = cv2.moments(self.c)
        self.centroid_x = int(M['m10']/M['m00'])
        self.centroid_y = int(M['m01']/M['m00'])
        self.height = self.get_line(theta=self.get_slope(points), x=self.centroid_x, y=self.centroid_y)
        return self.height, self.combined

    def give_prependicular_axis(self):
        """ Iterates through the symmetry axis and
            marks the prependicular axis with longest length
            returns dimension of line segment and 2-D image of (contour + line segment)
        @param points: array of (X, Y) co-ordinates of symmetry axis
        @param height: dimension of line segment
        @param combined: 2-D image of contour and symmetry axis marked
        """

        points = self.get_symm_line_points()
        m = self.get_slope(points) + (np.pi/2)

        for i in range(0, np.shape(points)[0]):
            y = points[i][0]
            x = points[i][1]
            temp_combined = np.zeros(self.frame_threshold.shape, dtype="uint8")
            temp = self.get_line(theta=m, x=x, y=y, check=False)
            if temp > self.width:
                self.width = temp
                y_final = y
                x_final = x
        self.width = self.get_line(theta=m, x=x_final, y=y_final)
        return self.width, self.combined

    def get_axis_by_symmetry(self):
        """ Marks the symmetry axis and prependicular axis with longest intercept
        @param centroid: True if want symmetric axis through centroid. Default value is False
        @param width: width of the fruit, in pixels
        @param height: height of the fruit, in pixels
        @param combined: 2-D image of contour and both axes marked
        Second part rotates the image, and finds the optimal axis of symmetry
        @param self.centroid_x: x-coord of contour
        @param self.centroid_y: y-coord of contour
        """
        if self.centroid_flag == True:
            self.height , self.combined = self.give_symmteric_axis_centroid()
        else:
            self.height , self.combined = self.give_symmteric_axis()
        self.width , self.combined = self.give_prependicular_axis()
        return self.width, self.height, self.combined


#-------------------------------------------------------------------

class rotateImage():

    def __init__(self, image=None):
        """ Initializes the code, driver code for rotating the image and finding most 
            optimal axis of symmetry through centroid
        @param image: 3-channel image of Fruit / onion
        """
        self.image = image
        self.min_area_rect = True
    
    def ratio(self, area1, area2):
        """ Returns ratio of areas of both contours
        @param area1: area of upper half of centroid
        @param area2: area of another half of centroid
        """
        temp = min(area1, area2) / max(area1, area2)
        return abs(1-temp)
    
    def get_axis_by_rotation(self):
        """ rotates the image, invokes the FruitsShape class and drives the code
            The image would be rotated around centroid of the contour
            Borders are added to provide enough space for rotation
            return type of this function is width, height, final_image
        """
        self.image = cv2.copyMakeBorder(self.image, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=0)
        fruit_shape0 = FruitShape(self.image)
        fruit_shape0.get_segmentation_mask()
        fruit_shape0.get_contour()

        centroid_x = fruit_shape0.centroid_x
        centroid_y = fruit_shape0.centroid_y

        finalRatio = 99
        (h, w) = self.image.shape[:2]
        final_rotated = self.image

        for i in range(0, 361, 5):
            M = cv2.getRotationMatrix2D((centroid_x, centroid_y), i, 1.0)
            rotated = cv2.warpAffine(self.image, M, (w, h))
            fruit_shape = FruitShape(rotated)
            fruit_shape.line_thickness = 3
            fruit_shape.get_segmentation_mask()
            fruit_shape.get_contour()
            length, temp_image = fruit_shape.give_symmteric_axis_centroid()
            cnts, _ = cv2.findContours(temp_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            if len(cnts) < 3:
                continue
            c1 = cnts[1]
            c2 = cnts[2]
            if self.min_area_rect:
                ((x1, y1), (w1, h1), angle1) = cv2.minAreaRect(c1)
                ((x2, y2), (w2, h2), angle2) = cv2.minAreaRect(c2)
                area1 = w1*h1
                area2 = w2*h2
            else:
                area1 = cv2.contourArea(c1)
                area2 = cv2.contourArea(c2)

            areaRatio = self.ratio(area1, area2)
            if areaRatio < finalRatio:
                finalRatio = areaRatio
                final_rotated = rotated

        fruit_shape1 = FruitShape(final_rotated)
        fruit_shape1.get_segmentation_mask()
        fruit_shape1.get_contour()

        return fruit_shape1.get_axis_by_symmetry()
