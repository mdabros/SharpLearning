using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public static class TensorExtensions
    {

        /// <summary>
        /// 
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T> func)
        {
            tensor.Data.Map(func);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="func"></param>
        public static void Map<T>(this Tensor<T> tensor, Func<T, T> func)
        {
            tensor.Data.Map(func);
        }
    }
}
