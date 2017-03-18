using System;
using System.Numerics;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.Providers.LinearAlgebra;
using SharpLearning.Containers.Tensors;
using SharpLearning.Neural.LayersNew;
using SharpLearning.Containers.Extensions;


namespace SharpLearning.Neural.Providers.DotNetOp
{
    /// <summary>
    /// 
    /// </summary>
    public static class Dense
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Forward(Variable input,
            Variable weights, Variable bias,
            Variable output, Executor executor)
        {
            var src = executor.GetTensor(input);

            var w = executor.GetTensor(weights);
            var b = executor.GetTensor(bias).Data;

            var dst = executor.GetTensor(output);

            src.Multiply(w, dst);
            dst.AddRowWise(b, dst);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="output"></param>
        /// <param name="executor"></param>
        public static void Backward(Variable input,
            Variable weights, Variable bias,
            Variable output, Executor executor)
        {
            var src = executor.GetTensor(input);
            var srcDiff = executor.GetGradient(input);

            var w = executor.GetTensor(weights);
            var wDiff = executor.GetGradient(weights);

            var bDiff = executor.GetGradient(bias).Data;           
            var dstDiff = executor.GetGradient(output);

            // calculate gradients
            src.TransposeThisAndMultiply(dstDiff, wDiff);
            dstDiff.SumColumns(bDiff);

            // calculate delta for next layer
            dstDiff.TransposeAndMultiply(w, srcDiff);
        }


        #region DotNet IMPL

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        /// <param name="output"></param>
        static void Forward(Tensor<float> input,
            Tensor<float> weights, Tensor<float> bias,
            Tensor<float> output)
        {
            if (output.DimensionCount != 2)
            {
                throw new ArgumentException($"output must be 2-dimensional, was: {output.DimensionCount}");
            }

            int IC = 0;

            if (input.DimensionCount == 4)
            {
                // 4D, IC is the product of the last 3 dimensions. flatten to 2D.
                IC = input.DimensionOffSets[0];
            }
            else
            {
                // assume 2D and use width
                IC = input.Dimensions[1]; 
            }

            var dst = output;
            var src = input;

            var w = weights;

            int MB = dst.Dimensions[0];
            int OC = dst.Dimensions[1];

            var dstData = dst.Data;
            var srcData = src.Data;
            var wData = weights.Data;
            var bData = bias.Data;

            Parallel.For(0, MB, mb =>
            {
                var srcBOffSet = src.DimensionOffSets[0] * mb;
                var dstBOffSet = dst.DimensionOffSets[0] * mb;

                for (int oc = 0; oc < OC; ++oc)
                {
                    var wCOffSet = w.DimensionOffSets[0] * oc;

                    float d = 0.0f;
                    if(Vector.IsHardwareAccelerated)
                    {
                        d = InnerLoopSimd(IC, srcData, wData, srcBOffSet, wCOffSet);
                    }
                    else
                    {
                        d = InnerLoop(IC, srcData, wData, srcBOffSet, wCOffSet);
                    }

                    // add bias
                    d += bData[oc];

                    var dstIndex = dstBOffSet + oc;
                    dstData[dstIndex] = d;
                }
            });
        }

        static float InnerLoopSimd(int IC, float[] srcData, float[] wData, int srcBOffSet, int wCOffSet)
        {
            var simdLength = Vector<float>.Count;
            var ic = 0;

            var d = 0f;

            for (ic = 0; ic <= IC - simdLength; ic += simdLength)
            {
                var vSrc = new Vector<float>(srcData, srcBOffSet + ic);
                var vW = new Vector<float>(wData, wCOffSet + ic);
                d += Vector.Dot(vSrc, vW);
            }

            for (; ic < IC; ++ic)
            {
                var srcIndex = srcBOffSet + ic;
                var wIndex = wCOffSet + ic;

                d += srcData[srcIndex] * wData[wIndex];
            }

            return d;
        }

        static float InnerLoop(int IC, float[] srcData, float[] wData, int srcBOffSet, int wCOffSet)
        {
            var d = 0.0f;
            for (int ic = 0; ic < IC; ++ic)
            {
                var srcIndex = srcBOffSet + ic;
                var wIndex = wCOffSet + ic;

                d += srcData[srcIndex] * wData[wIndex];
            }

            return d;
        }

        #endregion
    }
}
