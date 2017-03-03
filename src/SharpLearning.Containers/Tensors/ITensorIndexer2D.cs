using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ITensorIndexer2D<T>
    {
        /// <summary>
        /// 
        /// </summary>
        int H { get; }

        /// <summary>
        /// 
        /// </summary>
        int W { get; }

        /// <summary>
        /// 
        /// </summary>
        int NumberOfElements { get; }

        /// <summary>
        /// 
        /// </summary>
        TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        T At(int h, int w);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="value"></param>
        void At(int h, int w, T value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeH(int w, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeW(int h, Interval1D interval, T[] output);
    }
}