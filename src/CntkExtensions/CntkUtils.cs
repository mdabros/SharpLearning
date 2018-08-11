using System;
using System.Collections.Generic;
using CNTK;

namespace CntkExtensions
{
    /// <summary>
    /// 
    /// </summary>
    public static class CntkUtils
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public  static ParameterVector AsParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        public static CNTK.Value GetTensors(CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end, CNTK.DeviceDescriptor device)
        {
            var cpu_tensors = Util.get_minibatch_data_CPU(shape, src, indices, indices_begin, indices_end);
            var result = CNTK.Value.CreateBatch(shape, cpu_tensors, device, true);
            return result;
        }

        static float[] get_minibatch_data_CPU(CNTK.NDShape shape, float[] src, int[] indices, int indices_begin, int indices_end)
        {
            var num_indices = indices_end - indices_begin;
            var row_length = shape.TotalSize;
            var result = new float[num_indices];
            var row_index = 0;
            for (var index = indices_begin; index != indices_end; index++)
            {
                result[row_index++] = src[indices[index]];
            }
            return result;
        }
    }
}
