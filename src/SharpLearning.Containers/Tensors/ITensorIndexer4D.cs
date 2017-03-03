using SharpLearning.Containers.Views;
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
        int N { get; }

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
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        T At(int w, int h, int d, int n);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="value"></param>
        void At(int w, int h, int d, int n, T value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeN(int h, int d, int n, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeC(int w, int d, int n, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeH(int w, int h, int n, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="d"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeW(int w, int h, int d, Interval1D interval, T[] output);
    }
}
