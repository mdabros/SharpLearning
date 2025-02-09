using System.Collections.Generic;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.CsvRowTransforms;

/// <summary>
/// Interface for CsvRow transforms
/// </summary>
public interface ICsvRowTransformer
{
    /// <summary>
    /// Transforms a csv row
    /// </summary>
    /// <param name="rows"></param>
    /// <returns></returns>
    IEnumerable<CsvRow> Transform(IEnumerable<CsvRow> rows);
}
