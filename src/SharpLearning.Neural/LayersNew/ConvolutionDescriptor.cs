using System;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public struct ConvolutionDescriptor
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterCount;
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterH;
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterW;
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
        /// <param name="filterCount">Number of filters</param>
        /// <param name="filterH">The height of each filter</param>
        /// <param name="filterW">The width of each filter</param>
        /// <param name="strideH">The vertical stride of the filter</param>
        /// <param name="strideW">The horizontal stride of the filter</param>
        /// <param name="padH">Zero padding at the top and bottom</param>
        /// <param name="padW">Zero padding to the left and right</param>
        public ConvolutionDescriptor(int filterCount, int filterH, int filterW,
            int strideH, int strideW,
            int padH, int padW)
        {
            if (filterCount < 1)
            { throw new ArgumentException($"filterCount must be at least 1, was {filterCount}"); }
            if (filterH < 1)
            { throw new ArgumentException($"filterH must be at least 1, was {filterH}"); }
            if (filterW < 1)
            { throw new ArgumentException($"filterW must be at least 1, was {filterW}"); }
            if (strideH < 1)
            { throw new ArgumentException($"strideH must be at least 1, was {strideH}"); }
            if (strideW < 1)
            { throw new ArgumentException($"strideW must be at least 1, was {strideW}"); }
            if (padH < 0)
            { throw new ArgumentException($"padH must be at least 0, was {padH}"); }
            if (padW < 0)
            { throw new ArgumentException($"padW must be at least 0, was {padW}"); }

            FilterCount = filterCount;
            FilterH = filterH;
            FilterW = filterW;
            StrideH = strideH;
            StrideW = strideW;
            PadH = padH;
            PadW = padW;
        }
    }
}
