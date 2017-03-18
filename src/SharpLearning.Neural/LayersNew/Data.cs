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
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<float> GetOrAllocateTensor(Variable variable)
        {
            if (Tensor == null)
            {
                Tensor = new Tensor<float>(variable.Shape, DataLayout.RowMajor);
            }

            return Tensor;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<float> GetOrAllocateGradient(Variable variable)
        {
            if (Gradient == null)
            {
                Gradient = new Tensor<float>(variable.Shape, DataLayout.RowMajor);
            }

            return Gradient;
        }
    }
}
