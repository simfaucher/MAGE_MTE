using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Point = System.Drawing.Point;
using System.Runtime.InteropServices;
using Emgu.CV.Features2D;
using Emgu.CV.Util;

namespace Motion_Tracking_Engine
{
    /// <summary>
    /// Logique d'interaction pour MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        VideoCapture capture;
        public delegate void MonDelegate(Mat mat);

        Emgu.CV.Features2D.ORBDetector orbDetector;
        Emgu.CV.Features2D.Brisk briskDetector;
        Emgu.CV.Util.VectorOfKeyPoint refKeypoints;
        IOutputArray refDescriptors;
        Mat refImage;
        Mat frame;
        int MIN_MATCH_COUNT = 6;

        float tolerance = 0.5f;


        public MainWindow()
        {
            capture = new VideoCapture(); //create a camera capture
            InitializeComponent();

            #region EmguCVTest
            //String win1 = "Test Window"; //The name of the window
            //CvInvoke.NamedWindow(win1); //Create the window using the specific name

            //Mat img = new Mat(200, 400, DepthType.Cv8U, 3); //Create a 3 channel image of 400x200
            //img.SetTo(new Bgr(255, 0, 0).MCvScalar); // set it to Blue color

            ////Draw "Hello, world." on the image using the specific font
            //CvInvoke.PutText(
            //   img,
            //   "Hello, world",
            //   new Point(10, 80),
            //   FontFace.HersheyComplex,
            //   1.0,
            //   new Bgr(0, 255, 0).MCvScalar);


            //CvInvoke.Imshow(win1, img); //Show the image
            //CvInvoke.WaitKey(0);  //Wait for the key pressing event
            //CvInvoke.DestroyWindow(win1); //Destroy the window if key is pressed
            #endregion

            //ImageViewer viewer = new ImageViewer(); //create an image viewer
            //System.Windows.Forms.Application.Idle += new EventHandler(delegate (object sender, EventArgs e)
            //{  //run this until application closed (close button click on image viewer)
            //    viewer.Image = capture.QueryFrame(); //draw the image obtained from camera
            //});
            //viewer.ShowDialog(); //show the image viewer
            
            orbDetector = new Emgu.CV.Features2D.ORBDetector(50, 1.5f, 8, 50, patchSize:50);
            briskDetector = new Brisk();
            refKeypoints = new VectorOfKeyPoint();
            refDescriptors = new Mat();

            SetReference(capture.QueryFrame());
            Mat frame = new Mat();
            capture.Retrieve(frame);


            capture.ImageGrabbed += new EventHandler(delegate (object sender, EventArgs e)
            {

                Mat displayFrame = new Mat();
                capture.Retrieve(frame);
                /*
                frame = DetectAndComputeAndMatch(frame);
                /*/
                displayFrame = Match(frame);
                //*/
                // Permet d'afficher frame dans l'image Image
                Image.Dispatcher.BeginInvoke(System.Windows.Threading.DispatcherPriority.Normal, new MonDelegate(DrawImage), displayFrame);
            });
        }

        /// <summary>
        /// Delete a GDI object
        /// </summary>
        /// <param name="o">The poniter to the GDI object to be deleted</param>
        /// <returns></returns>
        [DllImport("gdi32")]
        private static extern int DeleteObject(IntPtr o);

        /// <summary>
        /// Convert an IImage to a WPF BitmapSource. The result can be used in the Set Property of Image.Source
        /// </summary>
        /// <param name="image">The Emgu CV Image</param>
        /// <returns>The equivalent BitmapSource</returns>
        public static BitmapSource ToBitmapSource(Mat image)
        {
            using (System.Drawing.Bitmap source = image.Bitmap)
            {
                IntPtr ptr = source.GetHbitmap(); //obtain the Hbitmap

                BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                    ptr,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());

                DeleteObject(ptr); //release the HBitmap
                return bs;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat"></param>
        private void DrawImage(Mat mat)
        {
            Image.Source = ToBitmapSource(mat); //draw the image obtained from camera
        }

        /// <summary>
        /// Event handler to detect when the button is clicked
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            SetReference(capture.QueryFrame()); //draw the image obtained from camera
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        //private Mat DetectAndComputeAndMatch(Mat image)
        //{
        //    orbDetector.DetectAndCompute(image, null, refKeypoints, refDescriptors, false);
        //    //briskDetector.DetectAndCompute(image, null, refKeypoints, refDescriptors, false);
        //    Mat ans = image.Clone();
        //    VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
        //    Emgu.CV.Features2D.BFMatcher matcher = new BFMatcher(DistanceType.L2Sqr);
        //    matcher.KnnMatch(refDescriptors, refDescriptors, matches, 2);
        //    Emgu.CV.Features2D.Features2DToolbox.DrawMatches(image, refKeypoints, image, refKeypoints, matches, ans, new MCvScalar(255, 0, 0), new MCvScalar(0, 0, 255));
        //    return ans;
        //}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        private Mat Match(Mat image)
        {
            VectorOfKeyPoint keypoints = new VectorOfKeyPoint();
            Mat descriptors = new Mat();

            List<PointF> srcPoints = new List<PointF>();
            List<PointF> dstPoints = new List<PointF>();


            orbDetector.DetectAndCompute(image, null, keypoints, descriptors, false);

            //Emgu.CV.Features2D.BFMatcher matcher = new BFMatcher(DistanceType.L2Sqr);
            Emgu.CV.Features2D.BFMatcher matcher = new BFMatcher(DistanceType.Hamming2);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();

            matcher.KnnMatch(descriptors, refDescriptors, matches, 2);
            //matcher.KnnMatch(descriptors, matches, 2, null);

            VectorOfVectorOfDMatch goodMatches = new VectorOfVectorOfDMatch();

            try
            {
                for (int i = 0; i < matches.Size; i++)
                {
                    // Ratio test as per Lowe's paper (usually 0.5 to 0.7)
                    if (matches[i][0].Distance < tolerance * matches[i][1].Distance)
                    {
                        goodMatches.Push(matches[i]);
                        PointF refPoint = refKeypoints[matches[i][0].QueryIdx].Point;
                        PointF dstPoint = keypoints[matches[i][0].TrainIdx].Point;
                        srcPoints.Add(new PointF(refPoint.X, refPoint.Y));
                        dstPoints.Add(new PointF(dstPoint.X, dstPoint.Y));
                    }
                }
            }
            catch(Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine(e.Source);
                Console.WriteLine(e.StackTrace);
            }


            if (goodMatches.Size >= MIN_MATCH_COUNT)
            {
                Mat M = CvInvoke.FindHomography(srcPoints.ToArray(), dstPoints.ToArray());

                var pts = new List<PointF>
                {
                    new PointF(0, 0),
                    new PointF(0, refImage.Height - 1),
                    new PointF(refImage.Width - 1, refImage.Height - 1),
                    new PointF(refImage.Width - 1, 0)
                };

                var dst = CvInvoke.PerspectiveTransform(pts.ToArray(), M);
                
                var polylinesPoints = new List<Point>();
                foreach (PointF pt in dst)
                {
                    polylinesPoints.Add(new Point((int)pt.X, (int)pt.Y));
                }

                CvInvoke.Polylines(image, polylinesPoints.ToArray(), true, new MCvScalar(255, 0, 0));
            }

            Mat ans = new Mat();
            Emgu.CV.Features2D.Features2DToolbox.DrawMatches(refImage, refKeypoints, image, keypoints, goodMatches, ans, new MCvScalar(0, 255, 0), new MCvScalar(0, 0, 255));
            return ans;
        }


        private void SetReference(Mat image)
        {
            refImage = image;
            orbDetector.DetectAndCompute(image, null, refKeypoints, refDescriptors, false);
        }

        private void CheckBox_Checked(object sender, RoutedEventArgs e)
        {
            capture.Start();
        }

        private void CheckBox_Unchecked(object sender, RoutedEventArgs e)
        {
            capture.Stop();
        }

        private void GetImageButton_Click(object sender, RoutedEventArgs e)
        {
            capture.QueryFrame();
        }
    }
}
