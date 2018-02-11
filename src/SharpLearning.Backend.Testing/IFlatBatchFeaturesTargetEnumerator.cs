namespace SharpLearning.Backend.Testing
{
    public interface IBatchEnumerator
    {
        int TotalBatchSize { get; }
        bool MoveNext();
        void Reset();
    }

    public interface ISplitFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> : IBatchEnumerator
    {
        // TODO: This doesn't scale well...! Need something else... dictionary?? FeatureTarget enum??
        TFeature[] CurrentFeatures();
        TTarget[] CurrentTargets();
    }


    public interface IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget>
    {
        int TotalBatchSize { get; }
        bool MoveNext();
        // TODO: This doesn't scale well...! Need something else... dictionary?? FeatureTarget enum??
        (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch();
        void Reset();
    }
}