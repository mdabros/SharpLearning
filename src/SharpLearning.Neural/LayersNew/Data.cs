using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Data<T>
    {
        /// <summary>
        /// 
        /// </summary>
        public Tensor<T> Tensor { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public Tensor<T> Gradient { get; set; }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetOrAllocateTensor(Variable variable)
        {
            if (Tensor == null)
            {
                Tensor = new Tensor<T>(variable.Shape, DataLayout.RowMajor);
            }

            return Tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="variable"></param>
        /// <returns></returns>
        public Tensor<T> GetOrAllocateGradient(Variable variable)
        {
            if (Gradient == null)
            {
                Gradient = new Tensor<T>(variable.Shape, DataLayout.RowMajor);
            }

            return Gradient;
        }
    }
}
