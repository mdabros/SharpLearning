using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Data
    {
        /// <summary>
        /// 
        /// </summary>
        public Tensor<float> Tensor { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Tensor<float> Gradient { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetOrAllocateTensor(TensorShape shape)
        {
            if (Tensor == null)
            {
                Tensor = new Tensor<float>(shape, DataLayout.RowMajor);
            }

            return Tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetOrAllocateGradient(TensorShape shape)
        {
            if (Gradient == null)
            {
                Gradient = new Tensor<float>(shape, DataLayout.RowMajor);
            }

            return Gradient;
        }
    }
}
