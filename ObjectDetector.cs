using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
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
        private OpenCvSharp.Dnn.Net? yoloNet;
        private List<string> classNames;
        private List<Scalar> classColors;
        private bool useAdvancedDetection = false;
        private bool useFeatureDetection = false;
        private bool useTracking = false;
        
        // Tracking için
        private Dictionary<int, KalmanFilter> trackers;
        private int nextTrackId = 0;
        private Dictionary<int, Rect> previousDetections;

        public ObjectDetector()
        {
            InitializeDetectors();
            InitializeClassNames();
            InitializeColors();
            trackers = new Dictionary<int, KalmanFilter>();
            previousDetections = new Dictionary<int, Rect>();
        }

        private void InitializeDetectors()
        {
            try
            {
                // Temel Haar Cascade sınıflandırıcıları
                faceCascade = new CascadeClassifier();
                faceCascade.Load("haarcascade_frontalface_alt.xml");
                
                eyeCascade = new CascadeClassifier();
                eyeCascade.Load("haarcascade_eye.xml");

                // HOG descriptor insan tespiti için
                hogDescriptor = new HOGDescriptor();
                hogDescriptor.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());

                // YOLO modeli yükleme (opsiyonel)
                try
                {
                    // YOLO modeli dosyaları varsa yükle
                    if (System.IO.File.Exists("yolov3.weights") && 
                        System.IO.File.Exists("yolov3.cfg") && 
                        System.IO.File.Exists("coco.names"))
                    {
                        yoloNet = OpenCvSharp.Dnn.CvDnn.ReadNetFromDarknet("yolov3.cfg", "yolov3.weights");
                        useAdvancedDetection = true;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"YOLO model yüklenemedi: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Detector initialization error: {ex.Message}");
            }
        }

        private void InitializeClassNames()
        {
            classNames = new List<string>
            {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            };
        }

        private void InitializeColors()
        {
            classColors = new List<Scalar>();
            var random = new Random(42); // Sabit seed ile tutarlı renkler
            
            for (int i = 0; i < classNames.Count; i++)
            {
                classColors.Add(new Scalar(
                    random.Next(0, 255),
                    random.Next(0, 255),
                    random.Next(0, 255)
                ));
            }
        }

        public Mat DetectObjects(Mat inputFrame)
        {
            Mat result = inputFrame.Clone();
            
            try
            {
                // Gelişmiş YOLO tespiti
                if (useAdvancedDetection && yoloNet != null)
                {
                    DetectObjectsYOLO(result);
                }
                else
                {
                    // Temel tespit yöntemleri
                    DetectFaces(result);
                    DetectPeople(result);
                    DetectColorObjects(result);
                }

                // Özellik tespiti
                if (useFeatureDetection)
                {
                    DetectFeatures(result);
                }

                // Nesne takibi
                if (useTracking)
                {
                    TrackObjects(result);
                }

                // Gelişmiş görüntü işleme (opsiyonel - yanlış tanıma yaratabilir)
                // ApplyAdvancedImageProcessing(result);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Detection error: {ex.Message}");
            }

            return result;
        }

        // YOLO ile nesne tespiti
        private void DetectObjectsYOLO(Mat frame)
        {
            if (yoloNet == null) return;

            using Mat blob = OpenCvSharp.Dnn.CvDnn.BlobFromImage(frame, 1.0 / 255.0, new OpenCvSharp.Size(416, 416), 
                new Scalar(0, 0, 0), true, false);
            
            yoloNet.SetInput(blob);
            
            var outputLayers = yoloNet.GetUnconnectedOutLayersNames();
            var outputs = new Mat[outputLayers.Length];
            yoloNet.Forward(outputs, outputLayers);

            var boxes = new List<Rect>();
            var confidences = new List<float>();
            var classIds = new List<int>();

            foreach (var output in outputs)
            {
                for (int i = 0; i < output.Rows; i++)
                {
                    var confidence = output.At<float>(i, 4);
                    if (confidence > 0.7) // Hassasiyeti artırdık (0.5'ten 0.7'ye)
                    {
                        var centerX = output.At<float>(i, 0) * frame.Width;
                        var centerY = output.At<float>(i, 1) * frame.Height;
                        var width = output.At<float>(i, 2) * frame.Width;
                        var height = output.At<float>(i, 3) * frame.Height;

                        var x = (int)(centerX - width / 2);
                        var y = (int)(centerY - height / 2);

                        boxes.Add(new Rect(x, y, (int)width, (int)height));

                        var maxClassId = 0;
                        var maxScore = 0.0f;
                        for (int j = 5; j < output.Cols; j++)
                        {
                            var score = output.At<float>(i, j);
                            if (score > maxScore)
                            {
                                maxScore = score;
                                maxClassId = j - 5;
                            }
                        }

                        confidences.Add(maxScore * confidence);
                        classIds.Add(maxClassId);
                    }
                }
            }

            // Non-maximum suppression - daha sıkı filtreleme
            var indices = new int[0];
            if (boxes.Count > 0)
            {
                OpenCvSharp.Dnn.CvDnn.NMSBoxes(boxes.ToArray(), confidences.ToArray(), 0.6f, 0.3f, out indices);
            }

            for (int i = 0; i < indices.Length; i++)
            {
                var idx = indices[i];
                var box = boxes[idx];
                var confidence = confidences[idx];
                var classId = classIds[idx];

                if (classId < classNames.Count)
                {
                    var color = classColors[classId];
                    Cv2.Rectangle(frame, box, color, 2);
                    
                    var label = $"{classNames[classId]}: {confidence:F2}";
                    var labelSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out var baseline);
                    
                    Cv2.Rectangle(frame, new Rect(box.X, box.Y - labelSize.Height - baseline, 
                        labelSize.Width, labelSize.Height + baseline), color, -1);
                    
                    Cv2.PutText(frame, label, new OpenCvSharp.Point(box.X, box.Y - baseline),
                        HersheyFonts.HersheySimplex, 0.5, Scalar.White, 1);
                }
            }
        }

        // Gelişmiş özellik tespiti (SIFT, SURF, ORB)
        private void DetectFeatures(Mat frame)
        {
            using Mat gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

            // SIFT özellik tespiti (OpenCvSharp4'te SIFT farklı pakette - geçici olarak devre dışı)
            // SIFT için ek paket gerekli: OpenCvSharp4.Extras
            // Şimdilik sadece ORB kullanıyoruz

            // ORB özellik tespiti
            var orb = OpenCvSharp.ORB.Create();
            var orbKeypoints = orb.Detect(gray);
            
            foreach (var kp in orbKeypoints)
            {
                Cv2.Circle(frame, new OpenCvSharp.Point((int)kp.Pt.X, (int)kp.Pt.Y), 2, Scalar.Cyan, -1);
            }
        }

        // Nesne takibi
        private void TrackObjects(Mat frame)
        {
            // Basit nesne takibi için Kalman Filter kullanımı
            // Bu örnekte sadece temel implementasyon gösteriliyor
            
            // Mevcut tespitleri al
            var currentDetections = new List<Rect>();
            
            // YOLO veya diğer tespit yöntemlerinden gelen sonuçları kullan
            // Burada basit bir örnek gösteriyoruz
            
            // Her tespit için takip ID'si ata
            foreach (var detection in currentDetections)
            {
                // Basit mesafe tabanlı takip
                int bestMatchId = -1;
                double bestDistance = double.MaxValue;
                
                foreach (var kvp in previousDetections)
                {
                    var distance = CalculateDistance(detection, kvp.Value);
                    if (distance < bestDistance && distance < 100) // 100 piksel threshold
                    {
                        bestDistance = distance;
                        bestMatchId = kvp.Key;
                    }
                }
                
                if (bestMatchId == -1)
                {
                    // Yeni nesne
                    bestMatchId = nextTrackId++;
                    trackers[bestMatchId] = new KalmanFilter(4, 2);
                }
                
                previousDetections[bestMatchId] = detection;
                
                // Takip ID'sini çiz
                Cv2.PutText(frame, $"ID: {bestMatchId}", 
                    new OpenCvSharp.Point(detection.X, detection.Y - 10),
                    HersheyFonts.HersheySimplex, 0.6, Scalar.Magenta, 2);
            }
        }

        private double CalculateDistance(Rect rect1, Rect rect2)
        {
            var center1 = new OpenCvSharp.Point(rect1.X + rect1.Width / 2, rect1.Y + rect1.Height / 2);
            var center2 = new OpenCvSharp.Point(rect2.X + rect2.Width / 2, rect2.Y + rect2.Height / 2);
            
            return Math.Sqrt(Math.Pow(center1.X - center2.X, 2) + Math.Pow(center1.Y - center2.Y, 2));
        }

        // Gelişmiş görüntü işleme teknikleri
        private void ApplyAdvancedImageProcessing(Mat frame)
        {
            // Gürültü azaltma
            using Mat denoised = new Mat();
            Cv2.FastNlMeansDenoisingColored(frame, denoised, 10, 10, 7, 21);
            
            // Kenar tespiti
            using Mat edges = new Mat();
            Cv2.Canny(denoised, edges, 50, 150);
            
            // Morfolojik işlemler
            using Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5));
            using Mat morphed = new Mat();
            Cv2.MorphologyEx(edges, morphed, MorphTypes.Close, kernel);
            
            // Sonuçları frame'e ekle (şeffaf overlay olarak)
            using Mat overlay = frame.Clone();
            overlay.SetTo(Scalar.Black, morphed);
            Cv2.AddWeighted(frame, 0.8, overlay, 0.2, 0, frame);
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
                minNeighbors: 5, // 3'ten 5'e artırdık (daha az yanlış pozitif)
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new OpenCvSharp.Size(50, 50) // 30'dan 50'ye artırdık
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
                groupThreshold: 3 // 2'den 3'e artırdık (daha az yanlış pozitif)
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
            DetectAdvancedColorObjects(frame);
            DetectShapes(frame);
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
                if (area > 1000) // 500'den 1000'e artırdık (daha büyük nesneler)
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
                if (area > 1000) // 500'den 1000'e artırdık (daha büyük nesneler)
                {
                    Rect boundingRect = Cv2.BoundingRect(contour);
                    Cv2.Rectangle(frame, boundingRect, Scalar.Blue, 2);
                    Cv2.PutText(frame, "Mavi Nesne", new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Blue, 2);
                }
            }
        }

        // Ayarlar ve kontrol metodları
        public void SetAdvancedDetection(bool enabled)
        {
            useAdvancedDetection = enabled;
        }

        public void SetFeatureDetection(bool enabled)
        {
            useFeatureDetection = enabled;
        }

        public void SetTracking(bool enabled)
        {
            useTracking = enabled;
        }

        public bool IsAdvancedDetectionAvailable()
        {
            return yoloNet != null;
        }

        // Performans optimizasyonu için GPU desteği
        public void EnableGPUAcceleration()
        {
            if (yoloNet != null)
            {
                try
                {
                    yoloNet.SetPreferableBackend(OpenCvSharp.Dnn.Backend.CUDA);
                    yoloNet.SetPreferableTarget(OpenCvSharp.Dnn.Target.CUDA);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"GPU acceleration failed: {ex.Message}");
                }
            }
        }

        // Gelişmiş renk tespiti
        private void DetectAdvancedColorObjects(Mat frame)
        {
            DetectRedObjects(frame);
            DetectBlueObjects(frame);
            DetectGreenObjects(frame);
            DetectYellowObjects(frame);
        }

        private void DetectGreenObjects(Mat frame)
        {
            using Mat hsv = new Mat();
            Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);

            using Mat lowerGreen = new Mat(1, 1, MatType.CV_8UC3, new Scalar(40, 50, 50));
            using Mat upperGreen = new Mat(1, 1, MatType.CV_8UC3, new Scalar(80, 255, 255));

            using Mat mask = new Mat();
            Cv2.InRange(hsv, lowerGreen, upperGreen, mask);

            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, 
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (OpenCvSharp.Point[] contour in contours)
            {
                double area = Cv2.ContourArea(OpenCvSharp.InputArray.Create(contour));
                if (area > 1000) // 500'den 1000'e artırdık (daha büyük nesneler)
                {
                    Rect boundingRect = Cv2.BoundingRect(contour);
                    Cv2.Rectangle(frame, boundingRect, Scalar.Green, 2);
                    Cv2.PutText(frame, "Yesil Nesne", new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Green, 2);
                }
            }
        }

        private void DetectYellowObjects(Mat frame)
        {
            using Mat hsv = new Mat();
            Cv2.CvtColor(frame, hsv, ColorConversionCodes.BGR2HSV);

            using Mat lowerYellow = new Mat(1, 1, MatType.CV_8UC3, new Scalar(20, 50, 50));
            using Mat upperYellow = new Mat(1, 1, MatType.CV_8UC3, new Scalar(30, 255, 255));

            using Mat mask = new Mat();
            Cv2.InRange(hsv, lowerYellow, upperYellow, mask);

            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, 
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (OpenCvSharp.Point[] contour in contours)
            {
                double area = Cv2.ContourArea(OpenCvSharp.InputArray.Create(contour));
                if (area > 1000) // 500'den 1000'e artırdık (daha büyük nesneler)
                {
                    Rect boundingRect = Cv2.BoundingRect(contour);
                    Cv2.Rectangle(frame, boundingRect, Scalar.Yellow, 2);
                    Cv2.PutText(frame, "Sari Nesne", new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
                }
            }
        }

        // Şekil tespiti
        private void DetectShapes(Mat frame)
        {
            using Mat gray = new Mat();
            Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
            
            using Mat blurred = new Mat();
            Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(5, 5), 0);
            
            using Mat edges = new Mat();
            Cv2.Canny(blurred, edges, 50, 150);
            
            Cv2.FindContours(edges, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy, 
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (OpenCvSharp.Point[] contour in contours)
            {
                double area = Cv2.ContourArea(OpenCvSharp.InputArray.Create(contour));
                if (area > 1000)
                {
                    var epsilon = 0.02 * Cv2.ArcLength(contour, true);
                    var approx = Cv2.ApproxPolyDP(contour, epsilon, true);
                    
                    var boundingRect = Cv2.BoundingRect(contour);
                    string shapeName = GetShapeName(approx.Length);
                    
                    Cv2.Rectangle(frame, boundingRect, Scalar.Orange, 2);
                    Cv2.PutText(frame, shapeName, new OpenCvSharp.Point(boundingRect.X, boundingRect.Y - 10),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Orange, 2);
                }
            }
        }

        private string GetShapeName(int vertices)
        {
            return vertices switch
            {
                3 => "Ucgen",
                4 => "Dortgen",
                5 => "Besgen",
                6 => "Altigen",
                _ => vertices > 6 ? "Cokgen" : "Bilinmeyen"
            };
        }

        public void Dispose()
        {
            faceCascade?.Dispose();
            eyeCascade?.Dispose();
            hogDescriptor?.Dispose();
            yoloNet?.Dispose();
            
            foreach (var tracker in trackers.Values)
            {
                tracker?.Dispose();
            }
            trackers.Clear();
            previousDetections.Clear();
        }
    }
}