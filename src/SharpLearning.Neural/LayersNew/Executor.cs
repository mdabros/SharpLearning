using System.Collections.Generic;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public class Executor
    {
        readonly Dictionary<TensorShape, Data> m_data;

        /// <summary>
        /// 
        /// </summary>
        public Executor()
        {
            m_data = new Dictionary<TensorShape, Data>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetTensor(TensorShape shape)
        {
            if(!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data();
            }

            return m_data[shape].GetOrAllocateTensor(shape);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape"></param>
        /// <returns></returns>
        public Tensor<float> GetGradient(TensorShape shape)
        {
            if (!m_data.ContainsKey(shape))
            {
                m_data[shape] = new Data();
            }

            return m_data[shape].GetOrAllocateGradient(shape);
        }
    }
}
