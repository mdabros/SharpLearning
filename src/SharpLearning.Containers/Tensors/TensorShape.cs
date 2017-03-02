using System;
using System.Linq;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public struct TensorShape
    {
        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// 
        /// </summary>
        public int NumberOfElements { get { return Elements(); } }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null) { throw new ArgumentNullException(); }
            Dimensions = dimensions;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        int Elements()
        {
            if (Dimensions.Length == 0)
                return 0;

            if (Dimensions.Sum() == 0)
                return 0;

            var elements = 1;
            foreach (var dimension in Dimensions)
            {
                elements *= dimension;
            }

            return elements;
        }
    }
}
