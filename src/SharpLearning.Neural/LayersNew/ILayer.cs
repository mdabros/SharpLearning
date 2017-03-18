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
        Variable Output { get; set; }

        /// <summary>
        /// Layer Input shape
        /// </summary>
        Variable Input { get; set; }

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
        /// <param name="inputVariable"></param>
        /// <param name="excecutor"></param>
        void Initialize(Variable inputVariable, Executor excecutor);
    }
}
