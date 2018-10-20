using CNTK;

namespace CntkCatalyst.LayerFunctions
{
    /// <summary>
    /// Layer operations for CNTK
    /// </summary>
    public static partial class Layers
    {
        public static Function BatchNormalization(this Function input,
            bool spatial,
            DeviceDescriptor device,
            DataType dataType,
            double initialScaleValue = 1,
            double initialBiasValue = 0,
            int bnTimeConst = 5000)
        {
            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialBiasValue, device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)initialScaleValue, device, "");
            var runningMean = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningCount = Constant.Scalar(0.0f, device);

            return CNTKLib.BatchNormalization(input, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
        }
    }
}
