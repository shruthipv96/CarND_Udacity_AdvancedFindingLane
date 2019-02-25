# ADVANCE LANE FINDING PROJECT

This is the second project and 

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib_undistort]: ./document_images/calib_undistort.png "Undistorted calibration image"
[test_undistort]:  ./document_images/test_undistort.png  "Undistorted test Image"
[sobel]:           ./document_images/sobel.png           "gradient image"
[color]:           ./document_images/color.png           "HLS image"
[threshold]:       ./document_images/threshold.png       "Threshold binary image"
[perspective]:     ./document_images/perspective.png     "Perspective check image"
[warped]:          ./document_images/warped.png          "Warped Threshold binary image"
[window]:          ./document_images/window.png          "Sliding Window image"
[final]:           ./document_images/final.png           "Final output"
[flowchart]:       ./document_images/flowchart.png       "Flowchart of the pipeline"

### The notebook for this project is [Project2_Shruthip.V](./Project2_Shruthip.V.ipynb)

Before proceeding to description of my pipeline, let me cover the Rubric Points.

---

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

Yes, I have submitted this [Project_2_Shruthip.V_README.md](./Project_2_Shruthip.V_README.md) for this project.

---

## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `third` code cell under the heading **`CAMERA CALIBRATION`** of the notebook mentioned above. 

I used the [images](../camera_cal) provided in the camera_cal folder for calibration. There were 20 images and it was mentioned in the Rubric that all were `(9x6)` chessboards. But I was not able to extract the corners from 3 images. So I ignored those and used the rest 17 images.*(I was able to identify the corners in rest of the images by changing the pattern. But I ignored to maintain uniformity).*

To calibrate camera, we need object points and image points. It is assumed that chessboard lies on the z=0 plane. Based on this, the object points are calculated as below and it remains the same unless the chessboard pattern is change.

```python
pattern    = (9,6)
objp       = np.zeros((pattern[1]*pattern[0],3),np.float32)
objp[:,:2] = np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)
```
This objp is added the object points whenever the corners are detected.

To calculate the image point, I used the OpenCV function `cv2.calibrateCamera()`. This returns the camera and distortion matrices and few more which I ignored.

In the lecture, the calibration was done with one image. It is known that the accuracy improves as the number of calibration images increase. But I observed that the `Camera Matrix` and `Distortion Matrix` calculated from each image was varying. So I took average of those respectively.

When I used these average values to undistort the input image,I observed the output was not as expected. I assumed that it was due to some unknown disturbance factor. So I started calculating that factor by trial and error and found to be approximately the following

```python
#Camera Matrix Correction Factor
mtx_factor = [ [15,0,16],
               [0,14,19],
               [0,0,17] ]

#Distortion Matrix Correction Factor
dist_factor = [[12,4,-0.8,0.8,1.2]]
```

I applied these factors on the average values correspondigly.When I used these **corrected factors** to undistort on the test image using `cv2.undistort()`, I found that the output was acceptable and looks like below:

![alt text][calib_undistort]

#### USAGE:
- Use `calibrate_camera()` function to calibrate the camera. The return is the corrected camera and distortion matrices.

---

## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the `fifth` code cell under the heading **`UNDISTORTION MODULE`** of the notebook mentioned above. 

Before using this module, we need to calibrate the camera as mentioned above.Once the camera matrix and the distortion matrix are obtained, we can undistort any image using `cv2.undistort()`. An example of the test image undistorted looks like below:(look for the chane at the bottom corners, the change is very minimum)

![alt text][test_undistort]

#### USAGE:
- Use `undistort(image)` function to use this module. Input is the distorted color image and output is the undistorted color image.

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the `sixth` code cell under the heading **`GRADIENT AND COLOR THRESHOLD MODULE`** of the notebook mentioned above.

I found the gradient of the image along x and y direction using `cv2.Sobel()` function. The output looked like below:

![alt text][sobel]

I also found that the magnitude and direction of the gradients. But I did not find so much help from these. So I ignored them.

I converted the image from RGB color space to HLS space using `cv2.cvtColor()` function. The individual channels look like below:

![alt text][color]

From the above two it is clear that the gradient along x direction and the S channel of HLS color space had a strong distinction for lanes than the other cases. Hence I combined these two transformations and produced a single thresholded binary image and it seems to work good for most of the cases. An example of this thresholded binary image looks like below:

![alt text][threshold]

#### USAGE:
- `abs_sobel_thresh()` returns binary gradient of the image.
- `S_color_threshold()` returns the binary image thresholded in S channel of HLS color space.
* Use `threshold_image(image,ksize = 3,grad_thresh=(20,100),color_thresh=(90,255))` function to use this module where image is the **undistorted image** ; ksize is the kernel size for the sobel function; grad_thresh is the threshold range for gradient thresholding and color_thresh is the threshold for S channel color thresholding. Return is the binary thresholded image of the input.

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the `fourth` code cell under the heading **`PERSCPECTIVE CALCULATION`** and `seventh` code cell under the heading **`WARP MODULE`** of the notebook mentioned above.

I split this into two modules because the perspective transform matrix needs to be calculated only once unless the camera orientation or position is changed whereas changing the perspective is done regularly.

**PERSPECTIVE CALCULATION:**

This module calculates the forward and reverse perspective transform matrices. To ease the calculation,I used a [straight lane line image](../test_images/straight_lines1.jpg). 

For calculating the perspective matrices, we need the source points and destination points (i.e) the points in one perspective to that of the points in another perspective. Assuming that the camera is in the same position and orientation, I have hardcoded these points as:

```python
    #Source points on the original image which looks like trapezoidal
    src = np.float32([(575,464),
                      (707,464), 
                      (258,682), 
                      (1049,682)])
    
    #Destination points on warped image which loooks like rectangle
    dst = np.float32([(250,0),
                      (w-275,0),
                      (250,h),
                      (w-275,h)])
```

which are the following points 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575 ,464      | 250 , 0       | 
| 707 ,464      | 1005, 0       |
| 258 ,682      | 250 ,720      |
| 1049,682      | 1005,720      |

Once the points are obtained, we can calculate the perspective transfrom matrices using `cv2.getPerspectiveTransform()` function. The calculation is done like this:
```python
    #Forward Transform
    M = cv2.getPerspectiveTransform(src, dst)
    #Inverse Transform
    Minv = cv2.getPerspectiveTransform(dst,src)
```
#### USAGE: 
- Use `calculate_perspective()` function to calculate the perspective matrices.

**WARP MODULE:**

Once the perspective matrices are calculated above, the perspective can be changed using `cv2.warpPerspective()` function.

To test the calculations are correct, I have drawn the points on the test image and the corresponding parallel lines on the warped image and it looks like below:

![alt text][perspective]

#### USAGE:
- Use `warp_image(image)` function to convert from *normal* view to *bird-eye* view
- Use `unwarp_image(image)` function to convert from *bird-eye* view to *normal* view

An example of the bird-eye of the thresholded binary image looks like below:

![alt text][warped]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the `eighth` code cell under the heading **`SLIDING WINDOW MODULE`** and `ninth` code cell under the heading **`polyfit_using_known_polyfit MODULE`** of the notebook mentioned above.

There are two ways to identify the lane line pixels and hence two modules.

**SLIDING WINDOW MODULE:**

The idea of this technique is to search the lane lines in a certain window than the whole image.

We divide the whole image as `12` windows with a window width of `80` pixels

* To decide the window position, we need to have a rough idea of the lane positions. Approximately the histogram at the border (i.e) height of the image gives a rough idea to start the window position. The positions of x where the values of histogram at base are maximum gives an idea of the base position of the lanes. We can start the window with those as centres of the window and keep sliding up. The code looks like below:
```python
# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
```

* Once we make a window at the base, we look for the pixel indices having `1` in this particular window. The code looks like below:
```python
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
```
* If the number of `1` in the particular window is more than a threshold, it means that there should be a shift in positions of the next window accordingly.The code looks like below:
```python
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```
* The above two steps are repeated until we slide through all the windows.

Once we find all the indices that belong to left lane and right lanes, we try to fit a polynomial with those points. The code look likes below:
```python
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

```

An example of the window and the fit found using this technique looks like below:

![alt text][window]

#### USAGE:
- Use `sliding_window_polyfit(image)` function to use this module. Input is the thresholded warped binary image and the return is the lane fits and the lane indices.

**polyfit_using_known_polyfit MODULE:**

We use sliding window technique to find the fit when we don't have much idea about the lane positions. But when we are driving, the lane lines don't change drastically, it will be a gradual change. Hence if we know the lane fits of one frame, then we can find use this to find the fits of the next frame. By this, we speed up our search process ,remove noise and shakiness to some extent and we can have a reliable fit to some extent.

The idea is to look for the lane pixels having value `1` around a margin of the known fit. The code looks like below,

```python
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
```

Once we find all the indices that belong to left lane and right lanes, we try to fit a polynomial with those points. The code look likes below:
     
```python
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

```
#### USAGE:
- Use `polyfit_using_known_polyfit(binary_warped,left_fit,right_fit)` function to use this module. Input is thresholded warped binary image, known left and right lane fits and the return is the lane fits and the lane indices.

**LINE class:**

The code for this step is contained in the `twelvth` code cell under the heading **`LINE class`** of the notebook mentioned above. 

We know that sliding window technique is for images where lane position are not known and polyfit_using_known_polyfit technique is for images where lane positions are roughly known. Hence to decide upon the lane finding techinque , we need to keep track of the results of the previous frame. So to store the results, a Line() class is defined which stores the results like,

- detected    -> To tell whether a lane line was detected in the frame.
- current_fit -> To store the current lane fit.

We know that the lane lines in the road does not change drastically and the change in fit will be minimum. So it is not advised to update the current_fit in the Line class everytime. Hence a method called `update_fit()` is added in the Line class, and it updates the current_fit only if the change in fit is minimum else it ignores the recent fit and uses the old fit. The code looks like,
```python
    #Update the fit if it is acceptable
    def update_fit(self,fit):
        #Check if the lanes are already detected
        if self.detected:
            #estimate the difference of available fit with the new fit
            diff = abs(fit - self.current_fit)
            #Check if the fit is acceptable and update accordingly
            if not ((diff[0] > 0.001) | 
                    (diff[1] > 1) |
                    (diff[2] > 100)):
                self.current_fit = fit 
                self.detected = True
            else:
                self.detected = False
        else:
            self.current_fit = fit
            self.detected = True
```

Once we know the details of the previous frame, we can decide upon the lane finding techinque. The code looks like below,*(This code is available in the `thirteenth` code cell of the notebook mentioned above under the heading **PROCESS IMAGE**)*

```python
    ## Check if there is a previous fit available
    if (left_l.detected & right_l.detected):
        #Load the old fit
        left_fit_old = left_l.current_fit
        right_fit_old= right_l.current_fit
        #Find the lane fit polynomials and lane points using polyfit_using_known_polyfit technique
        left_fit,right_fit,left_lane_inds,right_lane_inds = polyfit_using_known_polyfit(warped,left_fit_old,right_fit_old)     
       
    else:   
        #Find the lane fit polynomials and lane points using sliding window techinque
        left_fit,right_fit,left_lane_inds,right_lane_inds = sliding_window_polyfit(warped)
    
    #Update the current fit to the classes correspondingly       
    if len(left_fit) & len(right_fit):
        retl,left_fit = left_l.update_fit(left_fit)
        retr,right_fit = right_l.update_fit(right_fit)
    elif not len(left_fit):
        left_l.detected = False
    elif not len(right_fit):
        right_l.detected = False
```


### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the `tenth` code cell under the heading **`MEASUREMENT MODULE`** of the notebook mentioned above. 

We need to find the radius of curvature in metres.Hence we extract the indices of left and right lanes and multiply them with the factors that convert them to real world values. 

Assuming that the lane is about 30 meters long and 3.7 meters wide, the correspinding values in pixels are approximately 720 pixels and 700(approximately width of the image assuming that the lane completely occupies the image) pixels wide.
```python
#Conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
And with these factors, the new real world fit can be found by ,
```python
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
```
Once we know the fit, then the radius of curvature can be calculated using the formula,

R = ((1+(2Ay+B)^2)^1.5)/|2A|

The code looks like,
```python
    ##Calculate the radii of curvature in m
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

```

To calculate the vehicle distance from center,
- Assuming the camera is mounted at the center of the car, the position of the camera is `half of the width of the image` because the car occupies approximately along the whole width of image.
```python
    #Assuming that the camera is mounted at the center of the car
    car_mid = (width/2)
```
- To find the center of lane, I am extending the fits to the lower border of the image and finding the x value at the border. Once we find the the base positions of the lanes, then their average will give the middle of the lanes.
```python
    #Left lane base point
    left_base = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    #Right lane base point
    right_base = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    #Middle of the lanes at base point
    lane_mid =(left_base + right_base)/2
```
- Differnce in these values multiplied with xm_per_pix gives the difference in real world (in m).
```python
    #Distance of the car from middle of lane in m
    center_error = (car_mid - lane_mid)*xm_per_pix
```
Logic for finding direction,
- If the difference is negative, it means that middle of lane is at a position greater than middle of car then it means that the car is to the left of the center of lane.
- If the difference is positive, it means that middle of car is at a position greater than middle of lane then it means that the car is to the right of the center of lane.
- If the difference is zero, it means that car and lane are in same track.

An example measurements done for the image [test2.jpg](../test_images/test2.jpg)
```
Left Lane Radius of Curvature :  786.181868888 m
Right Lane Radius of Curvature:  410.377928281 m
Diff in centre of car and lane:  -0.190485024791 m
```

#### USAGE:
- Use `radius_of_curvature_and_centre_error(binary_warped,left_fit,right_fit,left_lane_inds,right_lane_inds)` function to use this module. Input is the threshold warped binary image, lane fits and the indices of the lanes and output is the radii of curvature of lanes and the distance of the car from the center of the lane.

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the `eleventh` code cell under the heading **`DRAWING TOOLS`** of the notebook mentioned above. 

- To draw polyline, I used `cv2.polylines()`.
- To draw polygon and fill the region, I used `cv2.fillPoly()`.
- To put text on image, I used `cv2.putText()`.

An example image which shows the lane drawn and the measured parameters written on the image.

![alt text][final]

#### USAGE:
- Use `create_four_sided_polygon(image,p1,p2,p3,p4,ignore_mask_color=255)` to draw a polygon with input points (p1,p2,p3,p4) and returns a polygon drawn on a black image.
- Use `draw_lane_on_original_image(undist,warped,left_fit,right_fit,Minv)` to draw a polygon on the image and unwarp it to normal space from bird-eye view.
- Use `put_data_on_image(image,radius_of_curvature,center_error)` to write the measured radius of curvature and the center of vehicle from middle of lane on the image as text.

---

## Pipeline (video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

With the pipeline implemented for processing the image worked pretty good for videos as expected and the [link to my video result](./output_videos/project_video.mp4) for the video [project_video.mp4](../project_video.mp4).
(The output is in ./output_videos/ folder)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code for the pipeline is contained in the `thirteenth` code cell under the heading **`PROCESS IMAGE`**.

With the explanation of the above modules in brief, I have shown the pipeline implementation as a flowchart below,

![alt text][flowchart]

**Observations**
- I observed that the lane detection was pretty good in shadow regions.
- I observed that the bumps in the road was very well identified and the output changed accordingly pretty good.

**Problems:**
- The pipeline fails when the video starts with a curve road. Because the histogram will be almost similar along the base line unless like a normal road where the lanes are distincitly seen in the histogram.
- When there is too much of detection other than the lane, there is high posiblity for the histogram peak to shift the base points over there.
- When there is a glass reflection in front of the camera, then there is some error in the calculation.

**Possible Improvement:**
- More robust method should be done to find the lanes on top of the histogram peaks. Like we may consider the base position of the lane from the previous frame and add a weight to this histogram peaks.
- We need to apply more transforms in some other colorspace or do some math on the gradient images to remove all the other detection.
- Along with the output, we need to check against some standards of the lane like width of the lane etc. to make sure that the detection is reliable.
