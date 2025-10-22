using System;
using System.Drawing;
using OpenCvSharp;
using DrawingPoint = System.Drawing.Point;
using DrawingSize = System.Drawing.Size;

namespace ObjectDetection
{
    public class ObjectDetector
    {
        private CascadeClassifier? faceCascade;
        private CascadeClassifier? eyeCascade;
        private HOGDescriptor? hogDescriptor;

        public ObjectDetector()
        {
            InitializeDetectors();
        }

        private void InitializeDetectors()
        {
            try
            {
                faceCascade = new CascadeClassifier();
                faceCascade.Load("haarcascade_frontalface_alt.xml");
                
                eyeCascade = new CascadeClassifier();
                eyeCascade.Load("haarcascade_eye.xml");

                hogDescriptor = new HOGDescriptor();
                hogDescriptor.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Detector initialization error: {ex.Message}");
            }
        }

        public Mat DetectObjects(Mat inputFrame)
        {
            Mat result = inputFrame.Clone();
            
            try
            {
                DetectFaces(result);
                DetectPeople(result);
                DetectColorObjects(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Detection error: {ex.Message}");
            }

            return result;
        }

        private void DetectFaces(Mat frame)
        {
            if (faceCascade == null) return;

            using Mat gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.EqualizeHist(gray, gray);

            Rect[] faces = faceCascade.DetectMultiScale(
                gray,
                scaleFactor: 1.1,
                minNeighbors: 3,
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new OpenCvSharp.Size(30, 30)
            );

            foreach (Rect face in faces)
            {
                Cv2.Rectangle(frame, face, Scalar.Green, 2);
                Cv2.PutText(frame, "Yuz", new OpenCvSharp.Point(face.X, face.Y - 10),
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Green, 2);
            }
        }

        private void DetectPeople(Mat frame)
        {
            if (hogDescriptor == null) return;

            Rect[] people = hogDescriptor.DetectMultiScale(
                frame,
                winStride: new OpenCvSharp.Size(8, 8),
                padding: new OpenCvSharp.Size(32, 32),
                scale: 1.05,
                groupThreshold: 2
            );

            foreach (Rect person in people)
            {
                Cv2.Rectangle(frame, person, Scalar.Blue, 2);
                Cv2.PutText(frame, "Insan", new OpenCvSharp.Point(person.X, person.Y - 10),
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Blue, 2);
            }
        }

        private void DetectColorObjects(Mat frame)
        {
            DetectRedObjects(frame);
            DetectBlueObjects(frame);
        }

        private void DetectRedObjects(Mat frame)
        {
            using Mat hsv = new Mat();
            Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);

            using Mat lowerRed1 = new Mat(1, 1, MatType.CV_8UC3, new Scalar(0, 50, 50));
            using Mat upperRed1 = new Mat(1, 1, MatType.CV_8UC3, new Scalar(10, 255, 255));
            using Mat lowerRed2 = new Mat(1, 1, MatType.CV_8UC3, new Scalar(170, 50, 50));
            using Mat upperRed2 = new Mat(1, 1, MatType.CV_8UC3, new Scalar(180, 255, 255));

            using Mat mask1 = new Mat();
            using Mat mask2 = new Mat();
            Cv2.InRange(hsv, lowerRed1, upperRed1, mask1);
            Cv2.InRange(hsv, lowerRed2, upperRed2, mask2);

            using Mat mask = new Mat();
            Cv2.BitwiseOr(mask1, mask2, mask);

            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, 
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (OpenCvSharp.Point[] contour in contours)
            {
                double area = Cv2.ContourArea(OpenCvSharp.InputArray.Create(contour));
                if (area > 500)
                {
                    Rect boundingRect = Cv2.BoundingRect(contour);
                    Cv2.Rectangle(frame, boundingRect, Scalar.Red, 2);
                    Cv2.PutText(frame, "Kirmizi Nesne", new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Red, 2);
                }
            }
        }

        private void DetectBlueObjects(Mat frame)
        {
            using Mat hsv = new Mat();
            Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);

            using Mat lowerBlue = new Mat(1, 1, MatType.CV_8UC3, new Scalar(100, 50, 50));
            using Mat upperBlue = new Mat(1, 1, MatType.CV_8UC3, new Scalar(130, 255, 255));

            using Mat mask = new Mat();
            Cv2.InRange(hsv, lowerBlue, upperBlue, mask);

            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, 
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (OpenCvSharp.Point[] contour in contours)
            {
                double area = Cv2.ContourArea(OpenCvSharp.InputArray.Create(contour));
                if (area > 500)
                {
                    Rect boundingRect = Cv2.BoundingRect(contour);
                    Cv2.Rectangle(frame, boundingRect, Scalar.Blue, 2);
                    Cv2.PutText(frame, "Mavi Nesne", new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Blue, 2);
                }
            }
        }

        public void Dispose()
        {
            faceCascade?.Dispose();
            eyeCascade?.Dispose();
            hogDescriptor?.Dispose();
        }
    }
}