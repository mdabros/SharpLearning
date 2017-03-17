namespace SharpLearning.Neural.Descriptors
{
    /// <summary>
    /// 
    /// </summary>
    public struct Conv2D
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterWidth;
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterHeight;
        /// <summary>
        /// 
        /// </summary>
        public readonly int FilterCount;
        /// <summary>
        /// 
        /// </summary>
        public readonly int StrideWidth;
        /// <summary>
        /// 
        /// </summary>
        public readonly int StrideHeight;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PadWidth;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PadHeight;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filterWidth"></param>
        /// <param name="filterHeight"></param>
        /// <param name="filterCount"></param>
        /// <param name="strideWidth"></param>
        /// <param name="strideHeight"></param>
        /// <param name="padWidth"></param>
        /// <param name="padHeight"></param>
        public Conv2D(int filterWidth, int filterHeight, int filterCount,
            int strideWidth, int strideHeight,
            int padWidth, int padHeight)
        {
            FilterWidth = filterWidth;
            FilterHeight = filterHeight;
            FilterCount = filterCount;
            StrideWidth = strideWidth;
            StrideHeight = strideHeight;
            PadWidth = padWidth;
            PadHeight = padHeight;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="filterWidth"></param>
        /// <param name="filterHeight"></param>
        /// <param name="filterCount"></param>
        /// <param name="strideWidth"></param>
        /// <param name="strideHeight"></param>
        /// <param name="borderMode"></param>
        public Conv2D(int filterWidth, int filterHeight, int filterCount, int strideWidth = 1, int strideHeight = 1,
            BorderMode borderMode = BorderMode.Valid)
            : this(filterWidth, filterHeight, filterCount, strideWidth, strideHeight,
                  ConvUtils.PaddingFromBorderMode(filterWidth, borderMode),
                  ConvUtils.PaddingFromBorderMode(filterHeight, borderMode))
        {
        }
    }
}

