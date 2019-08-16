namespace SharpLearning.Neural.Initializations
{
    /// <summary>
    /// 
    /// </summary>
    public struct FanInFanOut
    {
        /// <summary>
        /// The fan-in of the layer
        /// </summary>
        public readonly int FanIn;

        /// <summary>
        /// THe fan-out of the layer 
        /// </summary>
        public readonly int FanOut;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        public FanInFanOut(int fanIn, int fanOut)
        {
            FanIn = fanIn;
            FanOut = fanOut;
        }
    }
}
