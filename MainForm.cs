using System;
using System.Drawing;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using DrawingPoint = System.Drawing.Point;
using DrawingSize = System.Drawing.Size;
using DrawingTimer = System.Windows.Forms.Timer;

namespace ObjectDetection
{
    public partial class MainForm : Form
    {
        private VideoCapture? capture;
        private Mat? frame;
        private bool isCapturing = false;
        private ObjectDetector detector;
        private PictureBox pictureBox;
        private Button startButton;
        private Button stopButton;
        private ComboBox cameraComboBox;
        private Label statusLabel;
        private CheckBox advancedDetectionCheckBox;
        private CheckBox featureDetectionCheckBox;
        private CheckBox trackingCheckBox;
        private CheckBox gpuAccelerationCheckBox;
        private Label fpsLabel;
        private int frameCount = 0;
        private DateTime lastFpsUpdate = DateTime.Now;

        public MainForm()
        {
            InitializeComponent();
            detector = new ObjectDetector();
            InitializeCameraList();
        }

        private void InitializeComponent()
        {
            this.Text = "Gelişmiş Nesne Tanıma Sistemi";
            this.Size = new DrawingSize(1000, 700);
            this.StartPosition = FormStartPosition.CenterScreen;

            // Ana görüntü alanı
            pictureBox = new PictureBox
            {
                Size = new DrawingSize(640, 480),
                Location = new DrawingPoint(10, 10),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };

            // Kamera seçimi
            cameraComboBox = new ComboBox
            {
                Location = new DrawingPoint(10, 500),
                Size = new DrawingSize(150, 25),
                DropDownStyle = ComboBoxStyle.DropDownList
            };

            // Kontrol butonları
            startButton = new Button
            {
                Text = "Başlat",
                Location = new DrawingPoint(170, 500),
                Size = new DrawingSize(80, 25)
            };
            startButton.Click += StartButton_Click;

            stopButton = new Button
            {
                Text = "Durdur",
                Location = new DrawingPoint(260, 500),
                Size = new DrawingSize(80, 25),
                Enabled = false
            };
            stopButton.Click += StopButton_Click;


            // Durum ve FPS etiketleri
            statusLabel = new Label
            {
                Text = "Hazır",
                Location = new DrawingPoint(10, 530),
                Size = new DrawingSize(200, 20)
            };

            fpsLabel = new Label
            {
                Text = "FPS: 0",
                Location = new DrawingPoint(220, 530),
                Size = new DrawingSize(100, 20)
            };

            // Gelişmiş özellik kontrolleri
            advancedDetectionCheckBox = new CheckBox
            {
                Text = "Gelişmiş Tespit (YOLO)",
                Location = new DrawingPoint(670, 20),
                Size = new DrawingSize(200, 20),
                Checked = false
            };
            advancedDetectionCheckBox.CheckedChanged += AdvancedDetectionCheckBox_CheckedChanged;

            featureDetectionCheckBox = new CheckBox
            {
                Text = "Özellik Tespiti (SIFT/ORB)",
                Location = new DrawingPoint(670, 50),
                Size = new DrawingSize(200, 20),
                Checked = false
            };
            featureDetectionCheckBox.CheckedChanged += FeatureDetectionCheckBox_CheckedChanged;

            trackingCheckBox = new CheckBox
            {
                Text = "Nesne Takibi",
                Location = new DrawingPoint(670, 80),
                Size = new DrawingSize(200, 20),
                Checked = false
            };
            trackingCheckBox.CheckedChanged += TrackingCheckBox_CheckedChanged;

            gpuAccelerationCheckBox = new CheckBox
            {
                Text = "GPU Hızlandırma",
                Location = new DrawingPoint(670, 110),
                Size = new DrawingSize(200, 20),
                Checked = false
            };
            gpuAccelerationCheckBox.CheckedChanged += GpuAccelerationCheckBox_CheckedChanged;

            // Bilgi paneli
            var infoLabel = new Label
            {
                Text = "Gelişmiş Özellikler:",
                Location = new DrawingPoint(670, 140),
                Size = new DrawingSize(200, 20),
                Font = new Font("Arial", 10, FontStyle.Bold)
            };

            var infoText = new Label
            {
                Text = "• YOLO: 80 farklı nesne sınıfı\n• SIFT/ORB: Özellik noktaları\n• Takip: Kalman Filter\n• GPU: CUDA desteği",
                Location = new DrawingPoint(670, 160),
                Size = new DrawingSize(300, 100),
                Font = new Font("Arial", 8)
            };

            // Tüm kontrolleri forma ekle
            this.Controls.Add(pictureBox);
            this.Controls.Add(cameraComboBox);
            this.Controls.Add(startButton);
            this.Controls.Add(stopButton);
            this.Controls.Add(statusLabel);
            this.Controls.Add(fpsLabel);
            this.Controls.Add(advancedDetectionCheckBox);
            this.Controls.Add(featureDetectionCheckBox);
            this.Controls.Add(trackingCheckBox);
            this.Controls.Add(gpuAccelerationCheckBox);
            this.Controls.Add(infoLabel);
            this.Controls.Add(infoText);

            // Timer ayarları
            var timer = new DrawingTimer
            {
                Interval = 33 // ~30 FPS
            };
            timer.Tick += Timer_Tick;
            timer.Start();
        }

        private void InitializeCameraList()
        {
            cameraComboBox.Items.Clear();
            cameraComboBox.Items.Add("Kamera 0");
            cameraComboBox.Items.Add("Kamera 1");
            cameraComboBox.Items.Add("Kamera 2");
            cameraComboBox.SelectedIndex = 0;
        }

        private void StartButton_Click(object? sender, EventArgs e)
        {
            try
            {
                int cameraIndex = cameraComboBox.SelectedIndex;
                capture = new VideoCapture(cameraIndex);
                
                if (!capture.IsOpened())
                {
                    MessageBox.Show("Kamera açılamadı!", "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                isCapturing = true;
                startButton.Enabled = false;
                stopButton.Enabled = true;
                statusLabel.Text = "Kamera aktif";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Kamera hatası: {ex.Message}", "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void StopButton_Click(object? sender, EventArgs e)
        {
            isCapturing = false;
            capture?.Release();
            startButton.Enabled = true;
            stopButton.Enabled = false;
            statusLabel.Text = "Durduruldu";
            
            if (pictureBox.Image != null)
            {
                pictureBox.Image.Dispose();
                pictureBox.Image = null;
            }
        }

        private void Timer_Tick(object? sender, EventArgs e)
        {
            if (!isCapturing || capture == null) return;

            try
            {
                frame = new Mat();
                capture.Read(frame);

                if (!frame.Empty())
                {
                    var processedFrame = detector.DetectObjects(frame);
                    var bitmap = processedFrame.ToBitmap();
                    pictureBox.Image?.Dispose();
                    pictureBox.Image = bitmap;
                    
                    // FPS hesaplama
                    frameCount++;
                    var now = DateTime.Now;
                    if ((now - lastFpsUpdate).TotalSeconds >= 1.0)
                    {
                        var fps = frameCount / (now - lastFpsUpdate).TotalSeconds;
                        fpsLabel.Text = $"FPS: {fps:F1}";
                        frameCount = 0;
                        lastFpsUpdate = now;
                    }
                }
            }
            catch (Exception ex)
            {
                statusLabel.Text = $"Hata: {ex.Message}";
            }
        }

        // Event handler metodları
        private void AdvancedDetectionCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            detector.SetAdvancedDetection(advancedDetectionCheckBox.Checked);
            if (advancedDetectionCheckBox.Checked && !detector.IsAdvancedDetectionAvailable())
            {
                MessageBox.Show("YOLO modeli bulunamadı! Temel tespit yöntemleri kullanılacak.", "Uyarı", 
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                advancedDetectionCheckBox.Checked = false;
            }
        }

        private void FeatureDetectionCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            detector.SetFeatureDetection(featureDetectionCheckBox.Checked);
        }

        private void TrackingCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            detector.SetTracking(trackingCheckBox.Checked);
        }

        private void GpuAccelerationCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            if (gpuAccelerationCheckBox.Checked)
            {
                detector.EnableGPUAcceleration();
                MessageBox.Show("GPU hızlandırma etkinleştirildi.", "Bilgi", 
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }


        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            isCapturing = false;
            capture?.Release();
            frame?.Dispose();
            base.OnFormClosing(e);
        }
    }
}