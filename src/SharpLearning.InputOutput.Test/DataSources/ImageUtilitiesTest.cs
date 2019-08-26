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
            // TODO: Add in memory tests.
            var imageGetter = ImageSharpUtilities.GetImageLoader<Rgb24>(@"E:\mada\Dropbox\Images\image1.JPG");
            var transforms = new ImageTransformer<Rgb24>[] 
            {
                getter => ImageSharpUtilities.Resize(getter, 200, 200),
                getter => ImageSharpUtilities.Rotate(getter, 90)
            };

            var imageTransformer = ImageSharpUtilities.GetImageTransformer(imageGetter, transforms);

            using (var image = imageTransformer())
            {
                image.Save(@"E:\mada\Dropbox\Images\image1_test.JPG");

                var bytes = ImageSharpUtilities.ConvertImageToBytes(imageTransformer);
                var floats = ImageSharpUtilities.ConvertBytesToFloat(bytes);
            }
        }
    }
}
