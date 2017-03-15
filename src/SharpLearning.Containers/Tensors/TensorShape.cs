using System;
using System.Linq;

namespace SharpLearning.Containers.Tensors
{
    /// <summary>
    /// 
    /// </summary>
    public struct TensorShape : IEquatable<TensorShape>
    {
        /// <summary>
        /// 
        /// </summary>
        public TensorShape(params int[] dimensions)
        {
            if (dimensions == null) { throw new ArgumentNullException(); }
            Dimensions = dimensions;

            DimensionOffSets = new int[Dimensions.Length - 1];
            for (int i = 0; i < Dimensions.Length - 1; i++)
            {
                DimensionOffSets[i] = Dimensions.Skip(i + 1).Aggregate((x, y) => x * y);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// 
        /// </summary>
        public int[] DimensionOffSets { get; }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Elements(); } }

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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public bool Equals(TensorShape other)
        {
            if(Dimensions.Length != other.Dimensions.Length)
            {
                return false;
            }

            for (int i = 0; i < Dimensions.Length; i++)
            {
                if(Dimensions[i] != other.Dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator ==(TensorShape p1, TensorShape p2)
        {
            return p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="p2"></param>
        /// <returns></returns>
        public static bool operator !=(TensorShape p1, TensorShape p2)
        {
            return !p1.Equals(p2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (obj is ProbabilityPrediction)
                return Equals((ProbabilityPrediction)obj);
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked // Overflow is fine, just wrap
            {
                int hash = 17;
                hash = hash * 23 + Dimensions.GetHashCode();

                return hash;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            var dimensions = "TensorShape(";
            for (int i = 0; i < Dimensions.Length; i++)
            {
                dimensions += $"{Dimensions[i]}, ";
            }
            dimensions += ")";

            return dimensions;
        }
    }
}
