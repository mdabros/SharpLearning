namespace SharpLearning.Neural.Descriptors
{
    /// <summary>
    /// 
    /// </summary>
    public struct MaxPool2D
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int PoolWidth;
        /// <summary>
        /// 
        /// </summary>
        public readonly int PoolHeight;
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
        /// <param name="poolWidth"></param>
        /// <param name="poolHeight"></param>
        /// <param name="strideWidth"></param>
        /// <param name="strideHeight"></param>
        /// <param name="padWidth"></param>
        /// <param name="padHeight"></param>
        public MaxPool2D(int poolWidth, int poolHeight, 
            int strideWidth, int strideHeight, int padWidth, int padHeight)
        {
            PoolWidth = poolWidth;
            PoolHeight = poolHeight;
            StrideWidth = strideWidth;
            StrideHeight = strideHeight;
            PadWidth = padWidth;
            PadHeight = padHeight;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="poolWidth"></param>
        /// <param name="poolHeight"></param>
        /// <param name="strideWidth"></param>
        /// <param name="strideHeight"></param>
        /// <param name="borderMode"></param>
        public MaxPool2D(int poolWidth, int poolHeight,
            int strideWidth = 2, int strideHeight = 2, BorderMode borderMode = BorderMode.Valid)
            : this(poolWidth, poolHeight, strideWidth, strideHeight,
                  ConvUtils.PaddingFromBorderMode(poolWidth, borderMode),
                  ConvUtils.PaddingFromBorderMode(poolHeight, borderMode))
        {
        }
    }
}
