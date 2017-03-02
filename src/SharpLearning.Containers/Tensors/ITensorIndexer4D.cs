using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface ITensorIndexer4D<T>
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
        int DimZCount { get; }

        /// <summary>
        /// 
        /// </summary>
        int DimNCount { get; }

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
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        T At(int x, int y, int z, int n);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <param name="n"></param>
        /// <param name="value"></param>
        void At(int x, int y, int z, int n, T value);
    }
}
