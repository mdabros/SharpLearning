using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.XGBoost
{
    public enum BoosterType
    {
        /// <summary>
        /// Gradient boosted decision trees.
        /// </summary>
        GBTree,

        /// <summary>
        /// Gradient boosted linear models.
        /// </summary>
        GBLinear,

        /// <summary>
        /// DART: Dropouts meet Multiple Additive Regression Trees.
        /// http://xgboost.readthedocs.io/en/latest/tutorials/dart.html
        /// </summary>
        DART
    }
}
