using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Backend.Testing
{
    public interface IFunc<T, TResult> { TResult Invoke(T value); }

    public static class Transforms
    {
        public static void Transform<T, TResult, TFunc>(Span<T> src, TFunc func, Span<TResult> dst)
            where TFunc : IFunc<T, TResult>
        {
            if (src.Length != dst.Length) { throw new ArgumentException("lengths not equal"); }
            for (int i = 0; i < src.Length; i++)
            {
                dst[i] = func.Invoke(src[i]);
            }
        }
    }

    public class ConvertFlatBatchFeaturesTargetEnumerator<TFeature, TTarget, TFeatureResult, TTargetResult, TFeatureConverter, TTargetConverter>
        : IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget>
        where TFeatureConverter : IFunc<TFeature, TFeatureResult>
        where TTargetConverter : IFunc<TTarget, TTarget>
    {
        readonly IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> m_enumerator;
        readonly TFeatureConverter m_featureConverter;
        readonly TTargetConverter m_targetConverter;
        // NOTE: This reuses these!
        readonly TFeatureResult[] m_batchFeatures;
        readonly TTargetResult[] m_batchTargets;

        public ConvertFlatBatchFeaturesTargetEnumerator(IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget> enumerator)
        {
            m_enumerator = enumerator ?? throw new ArgumentNullException(nameof(enumerator));
            m_batchFeatures = new TFeatureResult[enumerator.TotalBatchSize];
            m_batchTargets = new TTargetResult[enumerator.TotalBatchSize];
        }

        public int TotalBatchSize => m_enumerator.TotalBatchSize;

        public bool MoveNext() => m_enumerator.MoveNext();

        public (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch()
        {
            var (fs, ts) = m_enumerator.CurrentBatch();
            // TODO: Do conversion
        }

        public void Reset() => m_enumerator.Reset();
    }

    // NOTE: This is specifically NOT an IEnumerator<T> for now, it could be perhaps
    public class FlatBatchFeaturesTargetEnumerator<TFeature, TTarget> : IFlatBatchFeaturesTargetEnumerator<TFeature, TTarget>
    {
        readonly FlatData<TFeature> m_features;
        readonly FlatData<TTarget> m_targets;
        readonly int m_count;
        readonly int m_batchSize;
        // NOTE: This reuses these!
        readonly TFeature[] m_batchFeatures;
        readonly TTarget[] m_batchTargets;
        int m_index = -1;

        // Can we always assume first dimension of shape is SampleCount??
        public FlatBatchFeaturesTargetEnumerator(FlatData<TFeature> features, FlatData<TTarget> targets, int batchSize)
        {
            m_features = features;
            m_targets = targets;
            m_batchSize = batchSize;
            var featuresSampleCount = m_features.Shape.SampleCount();
            var targetsSampleCount = m_targets.Shape.SampleCount();
            if (featuresSampleCount != targetsSampleCount)
            {
                throw new ArgumentException($"Features sample count {featuresSampleCount} not equal to target sample count {targetsSampleCount}");
            }
            m_count = featuresSampleCount;
            m_batchFeatures = new TFeature[m_features.Shape.FeatureSize() * batchSize];
            m_batchTargets = new TTarget[m_targets.Shape.FeatureSize() * batchSize];
        }

        public int TotalBatchSize => m_batchFeatures.Length;

        public bool MoveNext() => ++m_index < (m_count - m_batchSize + 1);

        // FUTURE: Make Span<T> version without memory copying, unfortunately no Backend supports span yet...
        public (TFeature[] batchFeatures, TTarget[] batchTargets) CurrentBatch()
        {
            Buffer.BlockCopy(m_features.Data, m_batchFeatures.Length * m_index, m_batchFeatures, 0, m_batchFeatures.Length);
            Buffer.BlockCopy(m_targets.Data, m_batchTargets.Length * m_index, m_batchTargets, 0, m_batchTargets.Length);
            return (m_batchFeatures, m_batchTargets);
        }
        
        public void Reset()
        {
            m_index = -1;
        }
    }
}
