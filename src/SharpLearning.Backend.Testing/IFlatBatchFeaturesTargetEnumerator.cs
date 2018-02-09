namespace SharpLearning.Backend.Testing
{
    public interface IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget>
    {
        int TotalBatchSize { get; }
        bool MoveNext();
        (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch();
        void Reset();
    }
}