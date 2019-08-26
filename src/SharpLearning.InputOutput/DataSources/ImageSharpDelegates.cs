using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Image transform delegate.
    /// </summary>
    /// <typeparam name="TPixel"></typeparam>
    /// <param name="imageGetter"></param>
    /// <returns></returns>
    public delegate ImageGetter<TPixel> ImageTransformer<TPixel>(ImageGetter<TPixel> imageGetter)
        where TPixel : struct, IPixel<TPixel>;

    /// <summary>
    /// Image getter delegate
    /// </summary>
    /// <typeparam name="TPixel"></typeparam>
    /// <returns></returns>
    public delegate Image<TPixel> ImageGetter<TPixel>()
        where TPixel : struct, IPixel<TPixel>;
}
