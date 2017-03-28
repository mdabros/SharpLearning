using System.Linq;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Variable
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="trainable"></param>
        /// <param name="preservable"></param>
        Variable(TensorShape shape, bool trainable = false, bool preservable = false)
        {
            Shape = shape;
            Trainable = trainable;
            Preservable = preservable;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="trainable"></param>
        /// <param name="preservable"></param>
        Variable(int[] dimensions, bool trainable = false, bool preservable = false)
            : this(new TensorShape(dimensions), trainable, preservable)
        { }

        /// <summary>
        /// 
        /// </summary>
        public TensorShape Shape { get; }

        /// <summary>
        /// 
        /// </summary>
        public bool Trainable { get; }

        /// <summary>
        /// 
        /// </summary>
        public bool Preservable { get; }

        /// <summary>
        /// 
        /// </summary>
        public int[] Dimensions { get { return Shape.Dimensions; } }

        /// <summary>
        /// 
        /// </summary>
        public int[] DimensionOffSets { get { return Shape.DimensionOffSets; } }

        /// <summary>
        /// 
        /// </summary>
        public int ElementCount { get { return Shape.ElementCount; } }

        /// <summary>
        /// 
        /// </summary>
        public int Rank { get { return Shape.Rank; } }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Variable Copy()
        {
            return new Variable(Dimensions.ToArray(), Trainable, Preservable);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable CreateTrainable(params int[] dimensions)
        {
            return new Variable(dimensions, true, true);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable CreatePreservable(params int[] dimensions)
        {
            return new Variable(dimensions, false, true);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable Create(params int[] dimensions)
        {
            return new Variable(dimensions, false);
        }
    }
}
