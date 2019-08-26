using System;
using System.Linq;
using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.DataSources;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace SharpLearning.InputOutput.Test.DataSources
{
    [TestClass]
    public class ImageUtilitiesTest
    {
        [TestMethod]
        public void Integration_Test()
        {
            // TODO: Add augmentation tests
            var imageDirectory = @"E:\mada\Dropbox\Images\køkken spas";

            var imageFilePaths = Directory.EnumerateFiles(imageDirectory);

            // images data loader.
            var random = new Random(Seed: 232);
            var imagesGetters = DataLoaders.EnumerateImages<Rgba32>(imageFilePaths)
                // Add augmentations
                .Select(i => i.Resize(200, 200)
                    .Brightness(1.0f, 3.0f, random)
                    .Flip(FlipMode.Horizontal, random)
                    .Skew(10, 10, random)
                    .Rotate(10, random)
                    .Zoom(1.1f, random));

            var index = 0;
            foreach (var imageGetter in imagesGetters)
            {
                using (var image = imageGetter())
                {
                    image.Save(Path.Combine(@"E:\test", index++.ToString() + ".JPG"));
                }
            }
        }
    }
}
