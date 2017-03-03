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
        int DimXCount { get; }

        /// <summary>
        /// 
        /// </summary>
        int DimYCount { get; }

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
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        T At(int x, int y);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="value"></param>
        void At(int x, int y, T value);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="y"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeX(int y, Interval1D interval, T[] output);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeY(int x, Interval1D interval, T[] output);
    }
}