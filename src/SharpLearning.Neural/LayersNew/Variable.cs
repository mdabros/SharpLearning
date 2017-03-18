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
        public Variable(TensorShape shape, bool trainable = false)
        {
            Shape = shape;
            Trainable = trainable;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="trainable"></param>
        public Variable(int[] dimensions, bool trainable = false)
            : this(new TensorShape(dimensions), trainable)
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
        public int DimensionCount { get { return Shape.DimensionCount; } }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Variable CreateTrainable(params int[] dimensions)
        {
            return new Variable(dimensions, true);
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
