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

        public MainForm()
        {
            InitializeComponent();
            detector = new ObjectDetector();
            InitializeCameraList();
        }

        private void InitializeComponent()
        {
            this.Text = "Nesne Tanıma Sistemi";
            this.Size = new DrawingSize(800, 600);
            this.StartPosition = FormStartPosition.CenterScreen;

            pictureBox = new PictureBox
            {
                Size = new DrawingSize(640, 480),
                Location = new DrawingPoint(10, 10),
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.Zoom
            };

            cameraComboBox = new ComboBox
            {
                Location = new DrawingPoint(10, 500),
                Size = new DrawingSize(150, 25),
                DropDownStyle = ComboBoxStyle.DropDownList
            };

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

            statusLabel = new Label
            {
                Text = "Hazır",
                Location = new DrawingPoint(350, 505),
                Size = new DrawingSize(200, 20)
            };

            this.Controls.Add(pictureBox);
            this.Controls.Add(cameraComboBox);
            this.Controls.Add(startButton);
            this.Controls.Add(stopButton);
            this.Controls.Add(statusLabel);

            var timer = new DrawingTimer
            {
                Interval = 33
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
                }
            }
            catch (Exception ex)
            {
                statusLabel.Text = $"Hata: {ex.Message}";
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