
using System;
namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// 1. The feature p used for the split
    //2. The split value xs;p+xs+1;p
    //3. The error on the left part of the split
    //4. The error on the right part of the split
    //5. The constant c1 that best ts the left region
    //6. The constant c2 that best ts the right region
    /// </summary>
    /// 
    [Serializable]
    public class GBMNode
    {
        public int FeatureIndex;
        public double SplitValue;
        public double LeftError;
        public double RightError;
        public double LeftConstant;
        public double RightConstant;
        public int Depth;
        public int LeftIndex = -1;
        public int RightIndex = -1;
        public int SampleCount;
    }
}
