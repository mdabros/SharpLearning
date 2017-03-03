namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ITensorIndexer3D<T>
    {
        /// <summary>
        /// 
        /// </summary>
        int C { get; }
        
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
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        T At(int c, int h, int w);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="value"></param>
        void At(int c, int h, int w, T value);
    }
}