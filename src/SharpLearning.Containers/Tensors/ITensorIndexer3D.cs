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
        int Width { get; }
        
        /// <summary>
        /// 
        /// </summary>
        int Height { get; }

        /// <summary>
        /// 
        /// </summary>
        int Depth { get; }

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
        /// <param name="d"></param>
        /// <returns></returns>
        T At(int w, int h, int d);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="value"></param>
        void At(int w, int h, int d, T value);
    }
}