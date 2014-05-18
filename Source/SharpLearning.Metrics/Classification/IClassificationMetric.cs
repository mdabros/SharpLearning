
namespace SharpLearning.Metrics.Classification
{
    public interface IClassificationMetric
    {
        double Error(double[] targets, double[] predictions);
        string ErrorString(double[] targets, double[] predictions);
    }
}
