
namespace SharpLearning.Metrics.Classification
{
    public interface IClassificationMetric<T>
    {
        double Error(T[] targets, T[] predictions);
        string ErrorString(T[] targets, T[] predictions);
    }
}
