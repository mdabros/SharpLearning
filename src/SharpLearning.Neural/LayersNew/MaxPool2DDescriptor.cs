using System;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public struct MaxPool2DDescriptor
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int PoolH;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PoolW;
        /// <summary>
        /// 
        /// </summary>
        public readonly int StrideH;
        /// <summary>
        /// 
        /// </summary>
        public readonly int StrideW;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PadH;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PadW;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="poolH">Height of the pooling window</param>
        /// <param name="poolW">Width of the pooling window</param>
        /// <param name="strideH">Pooling vertical stride</param>
        /// <param name="strideW">Pooling horizontal stride</param>
        /// <param name="padH">Size of vertical padding</param>
        /// <param name="padW">Size of horizontal padding</param>
        public MaxPool2DDescriptor(int poolH, int poolW, int strideH, int strideW,
                int padH, int padW)
        {
            if (poolH < 1)
            { throw new ArgumentException($"filterH must be at least 1, was {poolH}"); }
            if (poolW < 1)
            { throw new ArgumentException($"filterW must be at least 1, was {poolW}"); }
            if (strideH < 1)
            { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
            if (strideW < 1)
            { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
            if (padH < 0)
            { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
            if (padW < 0)
            { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

            PoolH = poolH;
            PoolW = poolW;
            StrideH = strideH;
            StrideW = strideW;
            PadH = padH;
            PadW = padW;
        }
    }
}
