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
    /// <param name="n"></param>
    /// <param name="c"></param>
    /// <param name="h"></param>
    /// <param name="w"></param>
    /// <returns></returns>
        T At(int n, int c, int h, int w);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="value"></param>
        void At(int n, int c, int h, int w, T value);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeN(int c, int h, int w, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="w"></param>
        /// <param name="d"></param>
        /// <param name="n"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeC(int n, int h, int w, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="w"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeH(int n, int c, int w, Interval1D interval, T[] output);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <param name="c"></param>
        /// <param name="h"></param>
        /// <param name="interval"></param>
        /// <param name="output"></param>
        void RangeW(int n, int c, int h, Interval1D interval, T[] output);
    }
}
