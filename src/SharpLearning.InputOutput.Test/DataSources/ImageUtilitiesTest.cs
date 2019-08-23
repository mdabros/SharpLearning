using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.DataSources;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.Test.DataSources
{
    [TestClass]
    public class ImageUtilitiesTest
    {
        [TestMethod]
        public void Integration_Test()
        {
            var imageGetter = ImageUtilities.GetImageLoader<Rgb24>(@"E:\mada\Dropbox\Images\image1.JPG");
            var transforms = new ImageTransformer<Rgb24>[] 
            {
                getter => ImageUtilities.Transforms.Resize(getter, 200, 200),
                getter => ImageUtilities.Transforms.Rotate(getter, 90)
            };

            var imageTransformer = ImageUtilities.GetImageTransformer(imageGetter, transforms);

            using (var image = imageTransformer())
            {
                image.Save(@"E:\mada\Dropbox\Images\image1_test.JPG");

                var bytes = ImageUtilities.ConvertImageToBytes(imageTransformer);
                var floats = ImageUtilities.ConvertBytesToFloat(bytes);
            }
        }
    }
}
