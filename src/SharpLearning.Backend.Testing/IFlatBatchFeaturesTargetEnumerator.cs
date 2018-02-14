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
        /// <summary>
        /// Size of batch in number of "samples"/observations.
        /// </summary>
        int BatchSize { get; }
        /// <summary>
        /// A single feature sample size in number of elements. NOT accounting for batch size.
        /// </summary>
        int FeaturesSize { get; }
        /// <summary>
        /// A single targets size in number of elements. NOT accounting for batch size.
        /// </summary>
        int TargetsSize { get; }

        bool MoveNext();
        // TODO: This doesn't scale well...! Need something else... dictionary?? FeatureTarget enum??
        (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch();
        void Reset();
    }
}