import cv2
import numpy as np
import panda3d.core as p3c

from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage

WEBCAM_RESIZE_RATIO = 0.5
N_CALIBRATION_IMAGES = 50
PATTERN_SIZE = (5, 3)
PATTERN_LENGTH = 26.26590886
MIN_DISTANCE = 20
VIDEO_PATH = "pose_dance_1.mp4"

USE_MIRROR_MODE = False

ARUCO_MARKER_ID = 3

# This class represents a simple application that uses a webcam to capture images and display them on the screen.
class MyApp(ShowBase):
    
    def __init__(self):
        # Call the constructor for the ShowBase class.
        ShowBase.__init__(self)

        # Check if the webcam is opened successfully.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Get the width and height of the webcam frame.
        self.imgW = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)*WEBCAM_RESIZE_RATIO)
        self.imgH = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*WEBCAM_RESIZE_RATIO)
        
        # Set the size of the window to the size of the webcam frame.
        winprops = p3c.WindowProperties()
        winprops.setSize(self.imgW, self.imgH)
        self.win.requestProperties(winprops)

        # Set the position of the camera to (0, -50, 0).
        self.cam.setPos(0, -50, 0)

        # Look at the point (0, 0, 0) with the camera.
        self.cam.lookAt(p3c.LPoint3f(0, 0, 0), p3c.LVector3f(0, 1, 0))

        # Create a texture object.
        self.tex = p3c.Texture()

        # Set up the texture object to be 2D, with the size of the webcam frame.
        self.tex.setup2dTexture(self.imgW, self.imgH, p3c.Texture.T_unsigned_byte, p3c.Texture.F_rgb)
        
        # Create an OnscreenImage object and set its image to the texture object.
        background = OnscreenImage(image=self.tex)
        
        # Reparent the OnscreenImage object to the render2dp node.
        background.reparentTo(self.render2dp)
        
        # Set the sort order of the display region in the render2dp node to -20.
        self.cam2dp.node().getDisplayRegion(0).setSort(-20)

        # Create an OnscreenText object and set its text to "Press '1' for calibration".
        self.textObject = OnscreenText(text="Press '1' for calibration", 
                                        pos=( -0.05, -0.95), 
                                        scale=(0.07, 0.07),
                                        fg=(1, 0.5, 0.5, 1), 
                                        align=p3c.TextNode.A_right,
                                        mayChange=1)
        
        # Reparent the OnscreenText object to the aspect2d node.
        self.textObject.reparentTo(self.aspect2d)

        # Bind the '1' key to the calibrateBegin() method.        
        self.accept('1', self.calibrateBegin)

        # Bind the '2' key to the calibrateEnd() method.
        self.accept('2', self.calibrateEnd)
 
        # Set the initial value for calibration.
        self.calibrateOn = False
        self.pattern_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
        self.pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
        self.pattern_points *= PATTERN_LENGTH

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        self.prev_center_corner = (0, 0)
        self.intrisic_mtx = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.dist_coefs = np.array([0,0,0,0])

        self.points3Ds = []
        self.points2Ds = []

        # Combine background update function to panda3d task manager.
        self.taskMgr.add(updateBg, 'video frame update')

        # Initial value for aruco marker
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.arucoParams = cv2.aruco.DetectorParameters()

        # Create a MovieTexture object named "Movie".
        self.myMovieTexture = p3c.MovieTexture("Movie")

        # Read the video file named VIDEO_PATH into the MovieTexture object.
        self.myMovieTexture.read(VIDEO_PATH)

        # Create a CardMaker object named "card".
        self.cm = p3c.CardMaker("card")

        # Set the UV range of the CardMaker object to the MovieTexture object.
        self.cm.setUvRange(self.myMovieTexture)

        # Get the aspect ratio of the MovieTexture object.
        aspect_ratio = self.myMovieTexture.getVideoWidth() / self.myMovieTexture.getVideoHeight()

        # Set the frame of the CardMaker object to (-1, 1, -1/aspect_ratio, 1/aspect_ratio).
        # It make video to keep there ratio.
        self.cm.setFrame(-1, 1, -1/aspect_ratio, 1/aspect_ratio)

        # Create a node and attach it to the render node.
        self.card_np = self.render.attachNewNode(self.cm.generate())

        # Set the texture of the node to the MovieTexture object.
        self.card_np.setTexture(self.myMovieTexture)

        # Stop the MovieTexture object.
        self.myMovieTexture.stop()

        # Get the 3D points from the card.
        self.get_3d_points_from_card()

        # Set video write
        self.out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, (self.imgW, self.imgH))

    # This function is called when the user presses the '1' key.
    def calibrateBegin(self):

        # Set the text of the text object to "calibrate On".
        self.textObject.text = "calibrate On"

        # Set the calibrateOn variable to True.
        self.calibrateOn = True

        # Reset the intrinsic matrix, distortion coefficients, 2D points, and 3D points.
        self.intrisic_mtx = None
        self.dist_coefs = None
        self.points2Ds = []
        self.points3Ds = []
        
    # This function is called when the user presses the '2' key.
    def calibrateEnd(self, text="calibrate Off"):

        # Set the text of the text object to the specified text.
        self.textObject.text = text

        # Set the calibrateOn variable to False.
        self.calibrateOn = False

    # This function gets the 3D points of the card node.
    def get_3d_points_from_card(self):

        # Get the bounding volume of the card node.
        bounds = self.card_np.get_bounds()

        # If the bounding volume exists and is not empty:
        if bounds is not None and not bounds.is_empty():  # bounding volume이 존재하는 경우
    
            # Get the center and radius of the bounding volume.
            center = bounds.get_approx_center()
            radius = bounds.get_radius()

            # Calculate the corner points of the bounding volume.
            corners = []
            for i in range(8):
                x = ((i & 1) << 1) - 1  # x 좌표 값 계산
                y = ((i & 2) >> 1) << 1 - 1  # y 좌표 값 계산
                z = ((i & 4) >> 2) << 1 - 1  # z 좌표 값 계산
                if y != 0:
                    continue
                # Calculate the corner point.
                # corner = p3c.Point3(x * radius, y * radius, z * radius) + center  
                xx = abs(x * radius)
                corners.append([-xx, 0, xx])
                corners.append([xx, 0, xx])
                corners.append([xx, 0, -xx])
                corners.append([-xx, 0, -xx])
                break
            
            # Save corners potins
            self.card_corners = np.array(corners)
        else:
            raise("Bounding volume does not exist.")

# This function calculates the distance between two points.
def distance_between_points(p1, p2):
    """
    Calculates the distance between two points.

    Args:
    p1: A NumPy array of shape (2,) representing the first point.
    p2: A NumPy array of shape (2,) representing the second point.

    Returns:
    The distance between the two points.
    """

    # Calculate the difference between the points.
    d = p2 - p1

    # Calculate the length of the difference vector.
    distance = np.linalg.norm(d)

    # Return the distance.
    return distance

# This function updates the background image.
def updateBg(task):
    """
    Updates the background image.

    Args:
    task: The task object.

    Returns:
    The task object.
    """

    # Read the next frame from the camera.
    success, frame = app.cap.read()
    if not success:
        raise("camera error!")
    
    # Resize the frame.
    frame = cv2.resize(frame, None, fx=WEBCAM_RESIZE_RATIO, fy=WEBCAM_RESIZE_RATIO)

    # Flip the frame horizontally for mirror mode.
    if USE_MIRROR_MODE:
        frame = cv2.flip(frame, 1)

    # If the camera is being calibrated and there are less than 50 points,
    # find the checkerboard corners in the frame.
    if app.calibrateOn == True and len(app.points2Ds) < N_CALIBRATION_IMAGES:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(img_gray, PATTERN_SIZE)

        # If the checkerboard corners were found,
        # draw them on the frame and get the center of the checkerboard.
        if found:
            
            # Get the corners of the checkerboard
            corners2 = cv2.cornerSubPix(img_gray, corners, PATTERN_SIZE, (-1, -1), app.criteria)
            center_corner = np.sum(corners2, axis=0).squeeze() / PATTERN_SIZE[0] / PATTERN_SIZE[1]

            # Get distance between current corners and prev corenrs.
            distance = distance_between_points(app.prev_center_corner, center_corner)                        
            if distance > MIN_DISTANCE and corners2[5][0][1] > corners2[0][0][1]:
                # Update progress bar.
                app.textObject.text = f"calibrating: {len(app.points2Ds):2d} / {N_CALIBRATION_IMAGES}"

                # Draw the corners on the image            
                cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners2, found)

                # Add new point pairs.
                app.points3Ds.append(app.pattern_points)
                app.points2Ds.append(corners2)

                # Save current center position of corners.
                app.prev_center_corner = center_corner
            
        # If the checkerboard corners were not found,
        # print a message.
        else:
            app.textObject.text = "checker board is not found!"
    # If there are N_CALIBRATION_IMAGES points,
    # calibrate the camera.
    elif app.calibrateOn == True and len(app.points2Ds) == N_CALIBRATION_IMAGES:
        app.calibrateEnd("Calibarating...")
        rms_err, intrisic_mtx, dist_coefs, rvecs, tvecs = \
            cv2.calibrateCamera(
                app.points3Ds, 
                app.points2Ds, 
                (app.imgH, app.imgH), 
                None, 
                None
            )
        print("\nRMS:", rms_err)
        app.calibrateEnd("Calibration is done!")
        app.intrisic_mtx = intrisic_mtx
        app.dist_coefs = dist_coefs

    # Undistort the frame if calibration is done.
    if app.calibrateOn == False and app.intrisic_mtx is not None:
        frame = cv2.undistort(frame, app.intrisic_mtx, app.dist_coefs, None, app.intrisic_mtx)
    
    #################
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, app.arucoDict, parameters=app.arucoParams)

    # if there is aruco marker for our tasks,
    # move camera position by using cv2.solvePnP().
    if ids is not None and ARUCO_MARKER_ID in ids:
        
        # Draw detected area of aruco marker.
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Get where our aruco mark is.
        index = np.where(ids == ARUCO_MARKER_ID)[0][0]

        # Get corners of aruco marker.
        marker_corners = corners[index].squeeze()

        # Get rvec, tvec from solvePnP().
        success, rvec, tvec = cv2.solvePnP(
            app.card_corners, 
            marker_corners, 
            app.intrisic_mtx, 
            app.dist_coefs, 
            flags=0
        )
        
        if success:
            # Get posion of camera for this frame.
            # this code is the same code with example.
            rmtx = cv2.Rodrigues(rvec)[0] # col-major
            
            matView = p3c.LMatrix4(rmtx[0][0], rmtx[1][0], rmtx[2][0], 0,
                                rmtx[0][1], rmtx[1][1], rmtx[2][1], 0,
                                rmtx[0][2], rmtx[1][2], rmtx[2][2], 0,
                                tvec[0], tvec[1], tvec[2], 1)

            matViewInv = p3c.LMatrix4()
            matViewInv.invertFrom(matView)

            cam_pos = matViewInv.xformPoint(p3c.LPoint3(0, 0, 0))
            cam_view = matViewInv.xformVec(p3c.LVector3(0, 0, 1))
            cam_up = matViewInv.xformVec(p3c.LVector3(0, -1, 0))

            app.cam.setPos(cam_pos)
            pos = app.cam.getPos()
            app.cam.setPos(pos[0], pos[1], pos[2])
            app.cam.lookAt(cam_pos + cam_view, cam_up)
            
            app.card_np.setPos(0, 0, 0)
            if not app.myMovieTexture.isPlaying():
                app.myMovieTexture.restart()
        else:
            app.myMovieTexture.stop()
    else:
        app.myMovieTexture.stop()

    # positive y goes down in openCV, so we must flip the y coordinates
    flip_frame = cv2.flip(frame, 0)

    # overwriting the memory with new frame
    app.tex.setRamImage(flip_frame)

    screenshot = app.win.getScreenshot() 
    arr = np.frombuffer(screenshot.getRamImageAs("RGB").getData(), dtype=np.uint8)
    arr = arr.reshape((screenshot.getYSize(), screenshot.getXSize(), 3))
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    flip_arr = cv2.flip(arr, 0)

    app.out.write(flip_arr)
    # return the task object.
    return task.cont

if __name__ == '__main__':
    app = MyApp()
    app.run()
    app.out.release()