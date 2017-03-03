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
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <returns></returns>
        T At(int w, int h);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="value"></param>
        void At(int w, int h, T value);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeH(int h, Interval1D interval, T[] output);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeW(int w, Interval1D interval, T[] output);
    }
}