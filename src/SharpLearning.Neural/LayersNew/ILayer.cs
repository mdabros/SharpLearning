using System.Collections.Generic;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public interface ILayerNew
    {
        /// <summary>
        /// Layer output shape
        /// </summary>
        TensorShape Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        TensorShape Input { get; set; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        void Forward(Executor executor);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="executor"></param>
        void Backward(Executor executor);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        void GetTrainableParameterShapes(List<TensorShape> parameters);
    }
}
