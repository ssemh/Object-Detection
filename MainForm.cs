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
        private Label fpsLabel;
        private Panel infoPanel;
        private Label titleLabel;
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
            this.Text = "Nesne Tanıma Sistemi";
            this.Size = new DrawingSize(800, 650);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(240, 240, 240);

            // Başlık etiketi
            titleLabel = new Label
            {
                Text = "Kamera Nesne Tanıma",
                Location = new DrawingPoint(20, 20),
                Size = new DrawingSize(760, 40),
                Font = new Font("Segoe UI", 20, FontStyle.Bold),
                ForeColor = Color.FromArgb(52, 73, 94),
                TextAlign = ContentAlignment.MiddleCenter
            };

            // Ana görüntü alanı - daha büyük ve merkezi
            pictureBox = new PictureBox
            {
                Size = new DrawingSize(760, 400),
                Location = new DrawingPoint(20, 70),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.Black
            };

            // Bilgi paneli - arka plan rengi ile
            infoPanel = new Panel
            {
                Location = new DrawingPoint(20, 480),
                Size = new DrawingSize(760, 120),
                BackColor = Color.FromArgb(255, 255, 255),
                BorderStyle = BorderStyle.FixedSingle
            };

            // Kamera seçimi
            var cameraLabel = new Label
            {
                Text = "Kamera:",
                Location = new DrawingPoint(20, 20),
                Size = new DrawingSize(80, 25),
                Font = new Font("Segoe UI", 10),
                ForeColor = Color.FromArgb(52, 73, 94)
            };

            cameraComboBox = new ComboBox
            {
                Location = new DrawingPoint(100, 20),
                Size = new DrawingSize(180, 28),
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 10)
            };

            // Kontrol butonları - daha büyük ve renkli
            startButton = new Button
            {
                Text = "▶ Başlat",
                Location = new DrawingPoint(300, 20),
                Size = new DrawingSize(120, 35),
                BackColor = Color.FromArgb(46, 204, 113),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 11, FontStyle.Bold),
                UseVisualStyleBackColor = false
            };
            startButton.FlatAppearance.BorderSize = 0;
            startButton.Click += StartButton_Click;

            stopButton = new Button
            {
                Text = "■ Durdur",
                Location = new DrawingPoint(440, 20),
                Size = new DrawingSize(120, 35),
                Enabled = false,
                BackColor = Color.FromArgb(231, 76, 60),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("Segoe UI", 11, FontStyle.Bold),
                UseVisualStyleBackColor = false
            };
            stopButton.FlatAppearance.BorderSize = 0;
            stopButton.Click += StopButton_Click;

            // Durum etiketi
            statusLabel = new Label
            {
                Text = "📹 Durum: Hazır",
                Location = new DrawingPoint(20, 70),
                Size = new DrawingSize(350, 30),
                Font = new Font("Segoe UI", 11),
                ForeColor = Color.FromArgb(46, 204, 113)
            };

            // FPS etiketi
            fpsLabel = new Label
            {
                Text = "⚡ FPS: 0",
                Location = new DrawingPoint(390, 70),
                Size = new DrawingSize(200, 30),
                Font = new Font("Segoe UI", 11),
                ForeColor = Color.FromArgb(52, 152, 219)
            };

            // Bilgi metni
            var infoText = new Label
            {
                Text = "💡 Kamera ile gerçek zamanlı nesne tespiti\n" +
                       "   Yüz, insan ve renkli nesneler otomatik algılanır",
                Location = new DrawingPoint(20, 105),
                Size = new DrawingSize(700, 50),
                Font = new Font("Segoe UI", 9),
                ForeColor = Color.FromArgb(127, 140, 141)
            };

            // Panel içine kontrolleri ekle
            infoPanel.Controls.Add(cameraLabel);
            infoPanel.Controls.Add(cameraComboBox);
            infoPanel.Controls.Add(startButton);
            infoPanel.Controls.Add(stopButton);
            infoPanel.Controls.Add(statusLabel);
            infoPanel.Controls.Add(fpsLabel);
            infoPanel.Controls.Add(infoText);

            // Tüm kontrolleri forma ekle
            this.Controls.Add(titleLabel);
            this.Controls.Add(pictureBox);
            this.Controls.Add(infoPanel);

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
                statusLabel.Text = "📹 Durum: Kamera aktif";
                statusLabel.ForeColor = Color.FromArgb(46, 204, 113);
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
            statusLabel.Text = "📹 Durum: Durduruldu";
            statusLabel.ForeColor = Color.FromArgb(231, 76, 60);
            
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
                        fpsLabel.Text = $"⚡ FPS: {fps:F1}";
                        frameCount = 0;
                        lastFpsUpdate = now;
                    }
                }
            }
            catch (Exception ex)
            {
                statusLabel.Text = $"⚠️ Hata: {ex.Message}";
                statusLabel.ForeColor = Color.FromArgb(231, 76, 60);
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